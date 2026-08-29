from __future__ import annotations

import numpy as np
from scipy import sparse


def ring_identity(size: int) -> np.ndarray:
    """Return a monomial identity using -1 for the zero polynomial."""
    identity = np.full((size, size), -1, dtype=np.int64)
    np.fill_diagonal(identity, 0)
    return identity


def ring_kron(left: np.ndarray, right: np.ndarray, ell: int) -> np.ndarray:
    """Kronecker product of monomial matrices over a cyclic ring."""
    output = np.full(
        (left.shape[0] * right.shape[0], left.shape[1] * right.shape[1]),
        -1,
        dtype=np.int64,
    )
    left_rows, left_columns = np.nonzero(left >= 0)
    right_rows, right_columns = np.nonzero(right >= 0)
    for i, j in zip(left_rows, left_columns, strict=True):
        for u, v in zip(right_rows, right_columns, strict=True):
            output[i * right.shape[0] + u, j * right.shape[1] + v] = (
                int(left[i, j]) + int(right[u, v])
            ) % ell
    return output


def circulant_shift(ell: int, exponent: int) -> sparse.csr_matrix:
    """Return the permutation matrix for multiplication by x**exponent."""
    columns = np.arange(ell)
    rows = (columns + exponent) % ell
    data = np.ones(ell, dtype=np.uint8)
    return sparse.csr_matrix((data, (rows, columns)), shape=(ell, ell))


def expand_ring_matrix(blocks: np.ndarray, ell: int) -> sparse.csr_matrix:
    """Expand a monomial block matrix into its binary circulant matrix."""
    expanded = [
        [None if int(exponent) < 0 else circulant_shift(ell, int(exponent)) for exponent in row]
        for row in blocks
    ]
    return sparse.bmat(expanded, format="csr", dtype=np.uint8)
