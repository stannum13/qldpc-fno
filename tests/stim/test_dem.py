import numpy as np
from scipy import sparse

from qldpc_fno.stim.dem import build_z_error_dem


def test_dem_matches_direct_binary_products() -> None:
    hx = sparse.csr_matrix([[1, 1, 0], [0, 1, 1]], dtype=np.uint8)
    logical_x = sparse.csr_matrix([[1, 1, 1]], dtype=np.uint8)
    dem = build_z_error_dem(hx, logical_x, error_rate=0.1)
    dets, obs, errors = dem.compile_sampler(seed=7).sample(64, return_errors=True)
    assert np.array_equal(dets, (errors @ hx.T.toarray()) % 2)
    assert np.array_equal(obs, (errors @ logical_x.T.toarray()) % 2)


def test_dem_rejects_invalid_error_rate() -> None:
    hx = sparse.csr_matrix([[1]], dtype=np.uint8)
    logical_x = sparse.csr_matrix([[1]], dtype=np.uint8)
    for error_rate in (-0.1, 0.0, 0.5, 1.0):
        try:
            build_z_error_dem(hx, logical_x, error_rate=error_rate)
        except ValueError:
            continue
        raise AssertionError(f"accepted invalid Bernoulli rate {error_rate}")
