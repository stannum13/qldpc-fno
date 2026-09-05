"""Generate an independent float64 oracle for the clipped AR grid transition."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.special import ndtr

_FIXTURE_PATH = Path("tests/identifiability/fixtures/clipped_grid_float64.json")
_CLIP = 1.2
_AR_COEFFICIENT = 0.97
_INNOVATION_STD = 0.08


def _grid(interior_cells: int) -> dict[str, object]:
    edges = np.linspace(-_CLIP, _CLIP, interior_cells + 1, dtype=np.float64)
    midpoints = 0.5 * (edges[:-1] + edges[1:])
    states = np.concatenate(([-_CLIP], midpoints, [_CLIP]))
    posterior = np.arange(1, states.size + 1, dtype=np.float64)
    posterior /= posterior.sum()

    standardized_edges = (edges[None, :] - _AR_COEFFICIENT * states[:, None]) / _INNOVATION_STD
    cdf_edges = ndtr(standardized_edges)
    transition_matrix = np.concatenate(
        (cdf_edges[:, :1], np.diff(cdf_edges, axis=1), 1.0 - cdf_edges[:, -1:]), axis=1
    )
    propagated = posterior @ transition_matrix
    return {
        "cell_edges": edges.tolist(),
        "interior_midpoints": midpoints.tolist(),
        "states": states.tolist(),
        "posterior": posterior.tolist(),
        "transition_matrix": transition_matrix.tolist(),
        "propagated": propagated.tolist(),
        "left_atom_mass": float(propagated[0]),
        "right_atom_mass": float(propagated[-1]),
    }


def _payload() -> dict[str, object]:
    return {
        "schema_version": 1,
        "implementation": "independent_scipy_cdf_exhaustive_matrix/v1",
        "clip": _CLIP,
        "ar_coefficient": _AR_COEFFICIENT,
        "innovation_std": _INNOVATION_STD,
        "interior_cells": [8, 16],
        "grids": {str(interior_cells): _grid(interior_cells) for interior_cells in (8, 16)},
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    rendered = json.dumps(_payload(), indent=2, sort_keys=True) + "\n"
    if args.check:
        return int(not _FIXTURE_PATH.is_file() or _FIXTURE_PATH.read_text() != rendered)
    _FIXTURE_PATH.parent.mkdir(parents=True, exist_ok=True)
    _FIXTURE_PATH.write_text(rendered)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
