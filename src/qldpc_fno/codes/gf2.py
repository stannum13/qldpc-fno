from __future__ import annotations

import numpy as np
from ldpc import mod2
from scipy import sparse


def gf2_rank(matrix: sparse.spmatrix) -> int:
    """Return the rank of a sparse binary matrix over GF(2)."""
    return int(mod2.rank(matrix, method="sparse"))


def gf2_product_is_zero(left: sparse.spmatrix, right_transpose: sparse.spmatrix) -> bool:
    """Return whether a sparse matrix product vanishes after reduction modulo two."""
    product = (left @ right_transpose).tocsr()
    return bool((product.data % 2 == 0).all())


def _row_to_bitset(matrix: sparse.csr_matrix, row: int) -> int:
    start, stop = matrix.indptr[row : row + 2]
    value = 0
    for column in matrix.indices[start:stop]:
        value ^= 1 << int(column)
    return value


def _add_to_bit_basis(value: int, pivots: dict[int, int]) -> bool:
    while value:
        pivot = value.bit_length() - 1
        if pivot not in pivots:
            pivots[pivot] = value
            return True
        value ^= pivots[pivot]
    return False


def quotient_basis(
    subspace: sparse.spmatrix,
    superspace: sparse.spmatrix,
) -> sparse.csr_matrix:
    """Select representatives extending a subspace to a supplied superspace."""
    subspace_csr = subspace.astype(np.uint8).tocsr()
    superspace_csr = superspace.astype(np.uint8).tocsr()
    if subspace_csr.shape[1] != superspace_csr.shape[1]:
        raise ValueError("subspace and superspace must have equal widths")
    pivots: dict[int, int] = {}
    for row in range(subspace_csr.shape[0]):
        _add_to_bit_basis(_row_to_bitset(subspace_csr, row), pivots)
    retained: list[int] = []
    for row in range(superspace_csr.shape[0]):
        if _add_to_bit_basis(_row_to_bitset(superspace_csr, row), pivots):
            retained.append(row)
    return superspace_csr[retained].tocsr()


def logical_x_basis(hx: sparse.spmatrix, hz: sparse.spmatrix) -> sparse.csr_matrix:
    """Return representatives of ker(Hz) modulo the X stabilizer row space."""
    hx_csr = hx.astype(np.uint8).tocsr()
    hz_csr = hz.astype(np.uint8).tocsr()
    if hx_csr.shape[1] != hz_csr.shape[1]:
        raise ValueError("Hx and Hz must have equal block length")
    if hz_csr.shape[0] == 0:
        kernel = sparse.eye(hz_csr.shape[1], format="csr", dtype=np.uint8)
    else:
        kernel = mod2.nullspace(hz_csr, method="sparse").astype(np.uint8).tocsr()
    logical = quotient_basis(hx_csr, kernel)
    expected = hx_csr.shape[1] - gf2_rank(hx_csr) - gf2_rank(hz_csr)
    if logical.shape[0] != expected:
        raise ValueError(f"expected {expected} logical X representatives, found {logical.shape[0]}")
    if not gf2_product_is_zero(logical, hz_csr.T):
        raise ValueError("logical X representatives do not commute with Hz")
    return logical
