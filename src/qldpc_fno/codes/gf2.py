from __future__ import annotations

from ldpc import mod2
from scipy import sparse


def gf2_rank(matrix: sparse.spmatrix) -> int:
    """Return the rank of a sparse binary matrix over GF(2)."""
    return int(mod2.rank(matrix, method="sparse"))


def gf2_product_is_zero(left: sparse.spmatrix, right_transpose: sparse.spmatrix) -> bool:
    """Return whether a sparse matrix product vanishes after reduction modulo two."""
    product = (left @ right_transpose).tocsr()
    return bool((product.data % 2 == 0).all())
