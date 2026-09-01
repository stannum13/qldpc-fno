"""Independent identity checks for the canonical accuracy-campaign code."""

from __future__ import annotations

import hashlib
from collections.abc import Mapping
from functools import lru_cache

import numpy as np
from scipy import sparse

from qldpc_fno.codes.lifted_product import build_self_lifted_product
from qldpc_fno.codes.seeds import PAPER_LP_3_7_16


def sparse_binary_sha256(matrix: sparse.spmatrix) -> str:
    """Hash canonical CSR structure and binary values independently of NPZ encoding."""
    canonical = matrix.astype(np.uint8).tocsr(copy=True)
    canonical.sum_duplicates()
    canonical.data %= 2
    canonical.eliminate_zeros()
    canonical.sort_indices()
    digest = hashlib.sha256()
    digest.update(np.asarray(canonical.shape, dtype="<i8").tobytes())
    digest.update(np.asarray(canonical.indptr, dtype="<i8").tobytes())
    digest.update(np.asarray(canonical.indices, dtype="<i8").tobytes())
    digest.update(np.asarray(canonical.data, dtype=np.uint8).tobytes())
    return digest.hexdigest()


@lru_cache(maxsize=1)
def _canonical_identity() -> tuple[dict[str, object], tuple[int, int], str, str]:
    code = build_self_lifted_product(PAPER_LP_3_7_16)
    metadata = {
        "ell": code.ell,
        "k": code.k,
        "n": code.n,
        "name": code.name,
    }
    return metadata, code.hx.shape, sparse_binary_sha256(code.hx), sparse_binary_sha256(code.hz)


def validate_campaign_code_identity(
    metadata: Mapping[str, object],
    hx: sparse.spmatrix,
    hz: sparse.spmatrix,
) -> None:
    """Validate metadata, dimensions, and content against a fresh paper-seed build."""
    expected_metadata, expected_shape, expected_hx_hash, expected_hz_hash = (
        _canonical_identity()
    )
    if any(metadata.get(key) != value for key, value in expected_metadata.items()):
        raise ValueError("campaign CLIs require the canonical lp_3_7_16 code metadata")
    if hx.shape != expected_shape or hz.shape != expected_shape:
        raise ValueError(f"campaign code matrix dimensions must both be {expected_shape}")
    if (
        sparse_binary_sha256(hx) != expected_hx_hash
        or sparse_binary_sha256(hz) != expected_hz_hash
    ):
        raise ValueError("campaign code matrix identity does not match PAPER_LP_3_7_16")
