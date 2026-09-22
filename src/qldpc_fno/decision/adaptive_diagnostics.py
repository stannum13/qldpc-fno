"""Development diagnostics for adaptive planar tensor inference."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np


def _probabilities(value: Sequence[float]) -> np.ndarray:
    probabilities = np.asarray(value, dtype=np.float64)
    if probabilities.shape != (4,):
        raise ValueError("logical-class probabilities must have shape (4,)")
    if np.any(~np.isfinite(probabilities)) or np.any(probabilities < 0):
        raise ValueError("logical-class probabilities must be finite and nonnegative")
    normalization = float(probabilities.sum())
    if not np.isfinite(normalization) or normalization <= 0:
        raise ValueError("logical-class probabilities must have positive finite mass")
    return probabilities / normalization


def probability_margin(probabilities: Sequence[float]) -> float:
    """Return the largest normalized probability minus the runner-up."""
    ordered = np.sort(_probabilities(probabilities))
    return float(ordered[-1] - ordered[-2])


def total_variation(left: Sequence[float], right: Sequence[float]) -> float:
    """Return total-variation distance between two four-class distributions."""
    return float(0.5 * np.abs(_probabilities(left) - _probabilities(right)).sum())


def jensen_shannon_divergence(left: Sequence[float], right: Sequence[float]) -> float:
    """Return Jensen-Shannon divergence in natural-log units."""
    left_array = _probabilities(left)
    right_array = _probabilities(right)
    midpoint = 0.5 * (left_array + right_array)

    def relative_entropy(probabilities: np.ndarray) -> float:
        positive = probabilities > 0
        return float(
            np.sum(probabilities[positive] * np.log(probabilities[positive] / midpoint[positive]))
        )

    return float(0.5 * (relative_entropy(left_array) + relative_entropy(right_array)))


def maximum_log_ratio_discrepancy(
    left: Sequence[float], right: Sequence[float]
) -> float | None:
    """Return maximum pairwise log-ratio error, or ``None`` when a mass is zero."""
    left_array = _probabilities(left)
    right_array = _probabilities(right)
    if np.any(left_array == 0) or np.any(right_array == 0):
        return None
    relative_logs = np.log(left_array) - np.log(right_array)
    return float(relative_logs.max() - relative_logs.min())


def _unavailable_spectral_summary() -> dict[str, object]:
    return {
        "available": False,
        "truncation_event_count": 0,
        "spectrum_count": 0,
        "maximum_discarded_squared_weight_fraction": None,
        "mean_spectral_entropy": None,
        "maximum_retained_rank": None,
        "estimated_arithmetic_flops": None,
    }


def summarize_spectra(action: Mapping[str, object] | None) -> dict[str, object]:
    """Aggregate spectra observed during an already charged tensor action."""
    if not isinstance(action, Mapping) or not isinstance(action.get("work"), Mapping):
        return _unavailable_spectral_summary()
    work = action["work"]
    assert isinstance(work, Mapping)
    events = work.get("truncation_events")
    estimated_flops = work.get("estimated_arithmetic_flops")
    if not isinstance(events, list) or type(estimated_flops) is not int or estimated_flops < 0:
        return _unavailable_spectral_summary()

    discarded: list[float] = []
    entropies: list[float] = []
    retained_ranks: list[int] = []
    for event in events:
        if not isinstance(event, Mapping) or not isinstance(event.get("spectral_summaries"), list):
            raise TypeError("tensor action has malformed truncation events")
        for spectrum in event["spectral_summaries"]:
            if not isinstance(spectrum, Mapping):
                raise TypeError("tensor action has malformed spectral summaries")
            discarded_value = spectrum.get("discarded_squared_weight_fraction")
            entropy_value = spectrum.get("spectral_entropy")
            rank_value = spectrum.get("retained_rank")
            if (
                not isinstance(discarded_value, (int, float))
                or isinstance(discarded_value, bool)
                or not np.isfinite(discarded_value)
                or not 0 <= discarded_value <= 1
                or not isinstance(entropy_value, (int, float))
                or isinstance(entropy_value, bool)
                or not np.isfinite(entropy_value)
                or entropy_value < 0
                or type(rank_value) is not int
                or rank_value < 0
            ):
                raise ValueError("tensor action has invalid spectral summary values")
            discarded.append(float(discarded_value))
            entropies.append(float(entropy_value))
            retained_ranks.append(rank_value)
    if not discarded:
        return {
            **_unavailable_spectral_summary(),
            "available": True,
            "truncation_event_count": len(events),
            "estimated_arithmetic_flops": estimated_flops,
        }
    return {
        "available": True,
        "truncation_event_count": len(events),
        "spectrum_count": len(discarded),
        "maximum_discarded_squared_weight_fraction": max(discarded),
        "mean_spectral_entropy": float(np.mean(entropies)),
        "maximum_retained_rank": max(retained_ranks),
        "estimated_arithmetic_flops": estimated_flops,
    }
