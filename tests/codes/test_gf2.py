import numpy as np
from scipy import sparse

from qldpc_fno.codes.gf2 import logical_x_basis, quotient_basis


def test_quotient_basis_adds_only_independent_rows() -> None:
    subspace = sparse.csr_matrix([[1, 1, 0]], dtype=np.uint8)
    superspace = sparse.csr_matrix([[1, 1, 0], [0, 0, 1]], dtype=np.uint8)
    logical = quotient_basis(subspace, superspace)
    assert logical.shape == (1, 3)
    assert np.array_equal(logical.toarray(), [[0, 0, 1]])


def test_logical_x_basis_is_quotient_of_kernel_by_x_stabilizers() -> None:
    hx = sparse.csr_matrix([[1, 1, 0], [0, 1, 1]], dtype=np.uint8)
    hz = sparse.csr_matrix((0, 3), dtype=np.uint8)
    logical = logical_x_basis(hx, hz)
    assert logical.shape == (1, 3)
    assert np.all((logical @ hz.T).data % 2 == 0)
