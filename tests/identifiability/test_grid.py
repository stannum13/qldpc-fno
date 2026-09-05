"""Tests for the clipped scalar AR(1) state-grid transition."""

from __future__ import annotations

import json
import subprocess
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
from scipy.special import ndtr

from qldpc_fno.identifiability.config import load_identifiability_config
from qldpc_fno.identifiability.grid import (
    ExperimentDeadlineExceeded,
    build_clipped_ar_grid,
    integrate_probability,
    transition_distribution,
    transition_from_point,
)

CONFIG_PATH = Path("configs/temporal_identifiability.json")
FIXTURE_PATH = Path("tests/identifiability/fixtures/clipped_grid_float64.json")


def _config():
    return load_identifiability_config(CONFIG_PATH)


def test_grid_has_exact_interior_edges_and_separate_clipping_atoms() -> None:
    config = _config()
    grid = build_clipped_ar_grid(config, interior_cells=config.grid.interior_cells)

    assert grid.interior_cells == 2_048
    assert grid.interior_midpoints.shape == (2_048,)
    assert grid.cell_edges.shape == (2_049,)
    assert grid.states.shape == (2_050,)
    assert grid.states[0] == -config.dynamics.clip
    assert grid.states[-1] == config.dynamics.clip
    assert grid.left_atom_index == 0
    assert grid.right_atom_index == 2_049
    assert grid.interior_midpoints[0] == pytest.approx(-1.2 + 2.4 / 4_096)
    assert grid.interior_midpoints[-1] == pytest.approx(1.2 - 2.4 / 4_096)
    np.testing.assert_array_equal(grid.cell_edges, np.linspace(-1.2, 1.2, 2_049))
    assert not np.any(grid.interior_midpoints == 0.0)


def test_round_zero_is_a_separate_exact_point_and_first_transition_uses_mean_zero() -> None:
    config = _config()
    grid = build_clipped_ar_grid(config, interior_cells=16)

    assert grid.initial_state_value == 0.0
    assert grid.initial_probability == config.dynamics.base_probability
    assert 0.0 not in grid.states

    actual = transition_from_point(grid, grid.initial_state_value)
    standardized_edges = grid.cell_edges / config.dynamics.innovation_std
    expected = np.concatenate(
        (
            [ndtr(standardized_edges[0])],
            np.diff(ndtr(standardized_edges)),
            [ndtr(-standardized_edges[-1])],
        )
    )
    np.testing.assert_allclose(actual, expected, rtol=0.0, atol=2e-15)
    base_logit = np.log(config.dynamics.base_probability / (1.0 - config.dynamics.base_probability))
    direct_mean_probability = np.dot(actual, 1.0 / (1.0 + np.exp(-(base_logit + grid.states))))
    assert integrate_probability(grid, actual) == pytest.approx(direct_mean_probability, abs=2e-15)


def test_point_transition_has_explicit_nonnegative_normalized_tail_mass() -> None:
    grid = build_clipped_ar_grid(_config(), interior_cells=8)
    actual = transition_from_point(grid, 0.95)

    assert actual.dtype == np.float64
    assert np.all(actual >= 0.0)
    assert actual.sum() == pytest.approx(1.0, abs=2e-15)
    assert actual[grid.left_atom_index] > 0.0
    assert actual[grid.right_atom_index] > actual[grid.left_atom_index]


def test_independent_fixture_script_regenerates_small_grids_without_production_imports() -> None:
    completed = subprocess.run(
        [sys.executable, "tools/generate_clipped_grid_fixture.py", "--check"],
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr
    fixture = json.loads(FIXTURE_PATH.read_text())
    assert fixture["implementation"] == "independent_scipy_cdf_exhaustive_matrix/v1"
    assert fixture["interior_cells"] == [8, 16]


@pytest.mark.parametrize("interior_cells", [8, 16])
def test_transition_matches_independent_exhaustive_float64_fixture(interior_cells: int) -> None:
    fixture = json.loads(FIXTURE_PATH.read_text())["grids"][str(interior_cells)]
    grid = build_clipped_ar_grid(_config(), interior_cells=interior_cells)
    posterior = np.asarray(fixture["posterior"], dtype=np.float64)

    actual = transition_distribution(grid, posterior)

    np.testing.assert_allclose(actual, fixture["propagated"], rtol=1e-13, atol=2e-15)
    assert actual.sum() == pytest.approx(1.0, abs=2e-15)
    assert np.all(actual >= 0.0)
    assert actual[0] == pytest.approx(fixture["left_atom_mass"], abs=2e-15)
    assert actual[-1] == pytest.approx(fixture["right_atom_mass"], abs=2e-15)


def test_transition_is_deterministic_and_rejects_invalid_posteriors() -> None:
    grid = build_clipped_ar_grid(_config(), interior_cells=16)
    posterior = np.linspace(1.0, 18.0, 18, dtype=np.float64)
    posterior /= posterior.sum()

    first = transition_distribution(grid, posterior)
    second = transition_distribution(grid, posterior)

    np.testing.assert_array_equal(first, second)
    with pytest.raises(ValueError, match="normalized"):
        transition_distribution(grid, posterior * 0.5)
    with pytest.raises(ValueError, match="nonnegative"):
        transition_distribution(grid, np.full(18, -1.0 / 18.0))


def test_transition_enforces_process_cpu_deadline(monkeypatch: pytest.MonkeyPatch) -> None:
    from qldpc_fno.identifiability import grid as grid_module

    grid = build_clipped_ar_grid(_config(), interior_cells=8)
    expired = replace(grid, process_cpu_deadline=0.0)
    monkeypatch.setattr(grid_module.time, "process_time", lambda: 1.0)

    with pytest.raises(ExperimentDeadlineExceeded, match="process CPU deadline"):
        transition_distribution(expired, np.full(10, 0.1))


def test_large_grid_uses_no_materialized_quadratic_transition_matrix() -> None:
    config = _config()
    grid = build_clipped_ar_grid(config, interior_cells=config.grid.doubled_interior_cells)

    assert grid.states.shape == (4_098,)
    assert not hasattr(grid, "transition_matrix")
    assert grid.source_batch_cells < grid.states.size
