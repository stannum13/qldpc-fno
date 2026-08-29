from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class LPSeed:
    """A monomial lifted-product seed and its reported code metadata."""

    name: str
    ell: int
    exponents: tuple[tuple[int, ...], ...]
    reported_n: int
    reported_k: int
    distance_upper_bound: int


PAPER_LP_3_7_16 = LPSeed(
    name="lp_3_7_16",
    ell=45,
    exponents=(
        (29, 21, 31, 15, 37, 25, 27),
        (13, 25, 19, 26, 11, 18, 29),
        (31, 2, 27, 32, 41, 41, 18),
    ),
    reported_n=2610,
    reported_k=744,
    distance_upper_bound=16,
)
