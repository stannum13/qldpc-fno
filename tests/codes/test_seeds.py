from qldpc_fno.codes.seeds import PAPER_LP_3_7_16


def test_paper_lp_3_7_16_seed_is_exact() -> None:
    assert PAPER_LP_3_7_16.name == "lp_3_7_16"
    assert PAPER_LP_3_7_16.ell == 45
    assert PAPER_LP_3_7_16.exponents == (
        (29, 21, 31, 15, 37, 25, 27),
        (13, 25, 19, 26, 11, 18, 29),
        (31, 2, 27, 32, 41, 41, 18),
    )
    assert PAPER_LP_3_7_16.reported_n == 2610
    assert PAPER_LP_3_7_16.reported_k == 744
    assert PAPER_LP_3_7_16.distance_upper_bound == 16
