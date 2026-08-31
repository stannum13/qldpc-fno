from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import sparse

from qldpc_fno.codes.gf2 import gf2_product_is_zero, gf2_rank
from qldpc_fno.codes.ring import circulant_shift, expand_ring_matrix, ring_identity, ring_kron
from qldpc_fno.codes.seeds import LPSeed


@dataclass(frozen=True, slots=True)
class CSSCode:
    """Sparse CSS checks and verified block metadata."""

    name: str
    ell: int
    hx: sparse.csr_matrix
    hz: sparse.csr_matrix
    n: int
    k: int
    distance_upper_bound: int


def _ring_dagger(matrix: np.ndarray, ell: int) -> np.ndarray:
    transposed = matrix.T
    return np.where(transposed >= 0, (-transposed) % ell, -1)


def build_self_lifted_product(seed: LPSeed) -> CSSCode:
    """Construct LP(A, A-dagger) using the convention cited by the paper."""
    a = np.asarray(seed.exponents, dtype=np.int64)
    r, n_a = a.shape
    a_dagger = _ring_dagger(a, seed.ell)
    hx_ring = np.hstack(
        [
            ring_kron(a, ring_identity(n_a), seed.ell),
            ring_kron(ring_identity(r), a_dagger, seed.ell),
        ]
    )
    hz_ring = np.hstack(
        [
            ring_kron(ring_identity(n_a), a, seed.ell),
            ring_kron(a_dagger, ring_identity(r), seed.ell),
        ]
    )
    hx = expand_ring_matrix(hx_ring, seed.ell).astype(np.uint8)
    hz = expand_ring_matrix(hz_ring, seed.ell).astype(np.uint8)
    n = int(hx.shape[1])
    k = n - gf2_rank(hx) - gf2_rank(hz)
    return CSSCode(
        name=seed.name,
        ell=seed.ell,
        hx=hx,
        hz=hz,
        n=n,
        k=k,
        distance_upper_bound=seed.distance_upper_bound,
    )


def _matrix_shift_equivariant(matrix: sparse.csr_matrix, ell: int) -> bool:
    row_blocks = matrix.shape[0] // ell
    column_blocks = matrix.shape[1] // ell
    shift = circulant_shift(ell, 1)
    row_shift = sparse.kron(sparse.eye(row_blocks, dtype=np.uint8), shift, format="csr")
    column_shift = sparse.kron(sparse.eye(column_blocks, dtype=np.uint8), shift, format="csr")
    difference = row_shift @ matrix - matrix @ column_shift
    difference.eliminate_zeros()
    return difference.nnz == 0


def validate_css(code: CSSCode) -> dict[str, object]:
    """Return machine-readable algebraic invariants for a CSS code."""
    hx_weights = np.diff(code.hx.indptr)
    hz_weights = np.diff(code.hz.indptr)
    commutes = gf2_product_is_zero(code.hx, code.hz.T)
    shift_equivariant = _matrix_shift_equivariant(code.hx, code.ell) and _matrix_shift_equivariant(
        code.hz, code.ell
    )
    hx_rank = gf2_rank(code.hx)
    hz_rank = gf2_rank(code.hz)
    matrix_n = code.hx.shape[1]
    same_block_length = matrix_n == code.hz.shape[1] == code.n
    computed_k = matrix_n - hx_rank - hz_rank
    dimension_matches = code.k == computed_k
    return {
        "name": code.name,
        "ell": code.ell,
        "n": code.n,
        "k": code.k,
        "distance_upper_bound": code.distance_upper_bound,
        "hx_shape": list(code.hx.shape),
        "hz_shape": list(code.hz.shape),
        "hx_rank": hx_rank,
        "hz_rank": hz_rank,
        "computed_k": computed_k,
        "dimension_matches_matrices": dimension_matches,
        "hx_row_weights": {"min": int(hx_weights.min()), "max": int(hx_weights.max())},
        "hz_row_weights": {"min": int(hz_weights.min()), "max": int(hz_weights.max())},
        "commutes": commutes,
        "ring_shift_equivariant": shift_equivariant,
        "valid": commutes and shift_equivariant and same_block_length and dimension_matches,
    }
