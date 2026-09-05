"""Bounded-memory transition for the clipped scalar AR(1) latent state."""

from __future__ import annotations

import math
import time
from dataclasses import dataclass

import numpy as np
from scipy.special import expit, ndtr

from qldpc_fno.identifiability.config import IdentifiabilityConfig

_MASS_TOLERANCE = 1e-12
_SOURCE_BATCH_CELLS = 128


class ExperimentDeadlineExceeded(RuntimeError):
    """Raised when a numerical transition outlives the process-CPU budget."""


def _readonly(array: np.ndarray, *, dtype: np.dtype | type | None = None) -> np.ndarray:
    result = np.array(array, dtype=dtype, copy=True, order="C")
    result.flags.writeable = False
    return result


@dataclass(frozen=True, slots=True)
class ClippedARGrid:
    """Finite grid plus separate atoms for the two clipping boundaries.

    ``states`` has one state for each atom and midpoint, in that order.  The
    exact round-zero value is intentionally *not* a grid state: with an even
    number of cells it lies on an interior cell edge and is represented by
    ``initial_state_value`` until its first transition.
    """

    interior_cells: int
    cell_edges: np.ndarray
    interior_midpoints: np.ndarray
    states: np.ndarray
    left_atom_index: int
    right_atom_index: int
    clip: float
    ar_coefficient: float
    innovation_std: float
    base_probability: float
    probability_clip: tuple[float, float]
    initial_state_value: float
    initial_probability: float
    process_cpu_deadline: float
    source_batch_cells: int
    mass_tolerance: float

    def __post_init__(self) -> None:
        if type(self.interior_cells) is not int or self.interior_cells <= 0:
            raise ValueError("interior_cells must be a positive integer")
        if self.left_atom_index != 0 or self.right_atom_index != self.interior_cells + 1:
            raise ValueError("boundary atoms must bracket every interior midpoint")
        if self.source_batch_cells <= 0:
            raise ValueError("source_batch_cells must be positive")
        if self.mass_tolerance <= 0.0 or not math.isfinite(self.mass_tolerance):
            raise ValueError("mass_tolerance must be positive and finite")
        if self.innovation_std <= 0.0 or not math.isfinite(self.innovation_std):
            raise ValueError("innovation_std must be positive and finite")
        if not all(
            math.isfinite(value)
            for value in (
                self.clip,
                self.ar_coefficient,
                self.base_probability,
                self.initial_state_value,
                self.initial_probability,
                self.process_cpu_deadline,
            )
        ):
            raise ValueError("grid scalar parameters must be finite")
        edges = np.asarray(self.cell_edges, dtype=np.float64)
        midpoints = np.asarray(self.interior_midpoints, dtype=np.float64)
        states = np.asarray(self.states, dtype=np.float64)
        if edges.shape != (self.interior_cells + 1,) or midpoints.shape != (
            self.interior_cells,
        ):
            raise ValueError("grid edges and midpoints must match interior_cells")
        if states.shape != (self.interior_cells + 2,):
            raise ValueError("states must contain two atoms plus every midpoint")
        if not np.all(np.isfinite(edges)) or not np.all(np.diff(edges) > 0.0):
            raise ValueError("cell_edges must be strictly increasing finite values")
        if edges[0] != -self.clip or edges[-1] != self.clip:
            raise ValueError("cell_edges must exactly meet the clipping boundaries")
        if not np.array_equal(midpoints, 0.5 * (edges[:-1] + edges[1:])):
            raise ValueError("interior_midpoints must be exact cell midpoints")
        if states[0] != -self.clip or states[-1] != self.clip or not np.array_equal(
            states[1:-1], midpoints
        ):
            raise ValueError("states must contain left atom, midpoints, and right atom")
        if not (-self.clip <= self.initial_state_value <= self.clip):
            raise ValueError("initial_state_value must be within clipping boundaries")
        object.__setattr__(self, "cell_edges", _readonly(edges, dtype=np.float64))
        object.__setattr__(self, "interior_midpoints", _readonly(midpoints, dtype=np.float64))
        object.__setattr__(self, "states", _readonly(states, dtype=np.float64))


def build_clipped_ar_grid(
    config: IdentifiabilityConfig, *, interior_cells: int, process_cpu_deadline: float
) -> ClippedARGrid:
    """Build a grid bound to the one process-CPU deadline for its enclosing run."""
    config.validate()
    if type(interior_cells) is not int or interior_cells <= 0:
        raise ValueError("interior_cells must be a positive integer")
    if not isinstance(process_cpu_deadline, (int, float, np.floating)) or not math.isfinite(
        float(process_cpu_deadline)
    ):
        raise ValueError("process_cpu_deadline must be finite")
    clip = config.dynamics.clip
    edges = np.linspace(-clip, clip, interior_cells + 1, dtype=np.float64)
    midpoints = 0.5 * (edges[:-1] + edges[1:])
    states = np.concatenate((np.array([-clip]), midpoints, np.array([clip])))
    return ClippedARGrid(
        interior_cells=interior_cells,
        cell_edges=edges,
        interior_midpoints=midpoints,
        states=states,
        left_atom_index=0,
        right_atom_index=interior_cells + 1,
        clip=clip,
        ar_coefficient=config.dynamics.ar_coefficient,
        innovation_std=config.dynamics.innovation_std,
        base_probability=config.dynamics.base_probability,
        probability_clip=config.dynamics.probability_clip,
        initial_state_value=0.0,
        initial_probability=config.dynamics.base_probability,
        process_cpu_deadline=float(process_cpu_deadline),
        source_batch_cells=_SOURCE_BATCH_CELLS,
        mass_tolerance=_MASS_TOLERANCE,
    )


def make_process_cpu_deadline(config: IdentifiabilityConfig) -> float:
    """Create the absolute CPU deadline once, at the start of an experiment run."""
    config.validate()
    return time.process_time() + config.runtime.process_cpu_seconds


def _check_deadline(grid: ClippedARGrid) -> None:
    if time.process_time() >= grid.process_cpu_deadline:
        raise ExperimentDeadlineExceeded("clipped-AR transition exceeded process CPU deadline")


def _normalize_distribution(grid: ClippedARGrid, distribution: np.ndarray) -> np.ndarray:
    if np.any(distribution < -grid.mass_tolerance):
        raise FloatingPointError("transition produced negative probability mass")
    result = np.maximum(distribution, 0.0)
    total = float(np.sum(result))
    if not math.isfinite(total) or abs(total - 1.0) > grid.mass_tolerance:
        raise FloatingPointError("transition failed probability-mass conservation")
    result /= total
    return result


def _normal_bin_probabilities(standardized_edges: np.ndarray) -> np.ndarray:
    """Return normal cell masses, using survival differences in the right tail."""
    lower = standardized_edges[..., :-1]
    upper = standardized_edges[..., 1:]
    probabilities = ndtr(upper) - ndtr(lower)
    positive_lower = lower > 0.0
    if np.any(positive_lower):
        probabilities[positive_lower] = (
            ndtr(-lower[positive_lower]) - ndtr(-upper[positive_lower])
        )
    return probabilities


def _point_transition(grid: ClippedARGrid, state: float) -> np.ndarray:
    mean = grid.ar_coefficient * state
    standardized_edges = (grid.cell_edges - mean) / grid.innovation_std
    result = np.empty(grid.states.shape, dtype=np.float64)
    result[grid.left_atom_index] = ndtr(standardized_edges[0])
    result[1:-1] = _normal_bin_probabilities(standardized_edges)
    result[grid.right_atom_index] = ndtr(-standardized_edges[-1])
    return _normalize_distribution(grid, result)


def transition_from_point(grid: ClippedARGrid, state: float) -> np.ndarray:
    """Transition an exact state point into the atom-plus-midpoint grid."""
    if not isinstance(grid, ClippedARGrid):
        raise TypeError("grid must be a ClippedARGrid")
    if not isinstance(state, (int, float, np.floating)) or not math.isfinite(float(state)):
        raise ValueError("state must be finite")
    if not -grid.clip <= float(state) <= grid.clip:
        raise ValueError("state must lie within clipping boundaries")
    _check_deadline(grid)
    result = _point_transition(grid, float(state))
    _check_deadline(grid)
    return result


def _validated_posterior(grid: ClippedARGrid, posterior: np.ndarray) -> np.ndarray:
    values = np.asarray(posterior, dtype=np.float64)
    if values.shape != grid.states.shape or not np.all(np.isfinite(values)):
        raise ValueError("posterior must be finite with one value per grid state")
    if np.any(values < 0.0):
        raise ValueError("posterior must be nonnegative")
    total = float(np.sum(values))
    if abs(total - 1.0) > grid.mass_tolerance:
        raise ValueError("posterior must be normalized")
    return values


def transition_distribution(grid: ClippedARGrid, posterior: np.ndarray) -> np.ndarray:
    """Propagate a grid posterior using streamed Gaussian-CDF source batches."""
    if not isinstance(grid, ClippedARGrid):
        raise TypeError("grid must be a ClippedARGrid")
    values = _validated_posterior(grid, posterior)
    _check_deadline(grid)
    result = np.zeros(grid.states.shape, dtype=np.float64)
    for start in range(0, grid.states.size, grid.source_batch_cells):
        _check_deadline(grid)
        stop = min(start + grid.source_batch_cells, grid.states.size)
        weights = values[start:stop]
        if not np.any(weights):
            continue
        means = grid.ar_coefficient * grid.states[start:stop, None]
        standardized_edges = (grid.cell_edges[None, :] - means) / grid.innovation_std
        result[grid.left_atom_index] += float(weights @ ndtr(standardized_edges[:, 0]))
        result[1:-1] += weights @ _normal_bin_probabilities(standardized_edges)
        result[grid.right_atom_index] += float(weights @ ndtr(-standardized_edges[:, -1]))
    _check_deadline(grid)
    return _normalize_distribution(grid, result)


def integrate_probability(grid: ClippedARGrid, distribution: np.ndarray) -> float:
    """Integrate physical error probability over a normalized latent distribution."""
    if not isinstance(grid, ClippedARGrid):
        raise TypeError("grid must be a ClippedARGrid")
    values = _validated_posterior(grid, distribution)
    base_logit = math.log(grid.base_probability / (1.0 - grid.base_probability))
    probabilities = np.clip(
        expit(base_logit + grid.states), grid.probability_clip[0], grid.probability_clip[1]
    )
    return float(values @ probabilities)
