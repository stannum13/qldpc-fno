import numpy as np

from qldpc_fno.codes.lifted_product import CSSCode, build_self_lifted_product, validate_css
from qldpc_fno.codes.seeds import PAPER_LP_3_7_16, LPSeed


def test_self_lifted_product_commutes_on_tiny_seed() -> None:
    seed = LPSeed("tiny", 5, ((0, 1),), 25, 9, 2)
    code = build_self_lifted_product(seed)
    assert code.hx.shape == (10, 25)
    assert code.hz.shape == (10, 25)
    product = code.hx @ code.hz.T
    assert np.all(product.data % 2 == 0)


def test_paper_code_shapes_dimension_and_weights() -> None:
    code = build_self_lifted_product(PAPER_LP_3_7_16)
    assert code.hx.shape == (945, 2610)
    assert code.hz.shape == (945, 2610)
    assert code.n == 2610
    assert code.k == 744
    assert set(np.diff(code.hx.indptr)) == {10}
    assert set(np.diff(code.hz.indptr)) == {10}

    checks = validate_css(code)
    assert checks["valid"] is True
    assert checks["commutes"] is True
    assert checks["ring_shift_equivariant"] is True
    assert checks["hx_row_weights"] == {"min": 10, "max": 10}


def test_validation_rejects_dimension_metadata_inconsistent_with_matrices() -> None:
    code = build_self_lifted_product(LPSeed("tiny", 5, ((0, 1),), 25, 9, 2))
    inconsistent = CSSCode(
        name=code.name,
        ell=code.ell,
        hx=code.hx,
        hz=code.hz,
        n=code.n,
        k=code.k + 1,
        distance_upper_bound=code.distance_upper_bound,
    )
    checks = validate_css(inconsistent)
    assert checks["computed_k"] == code.k
    assert checks["dimension_matches_matrices"] is False
    assert checks["valid"] is False
