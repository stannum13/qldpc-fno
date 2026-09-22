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


def _action_probabilities(action: object) -> np.ndarray | None:
    if not isinstance(action, Mapping):
        return None
    probabilities = action.get("probabilities")
    if not isinstance(probabilities, Sequence) or isinstance(probabilities, (str, bytes)):
        return None
    return _probabilities(probabilities)


def _selected_class(action: object) -> int | None:
    if not isinstance(action, Mapping):
        return None
    selected_class = action.get("selected_class")
    if type(selected_class) is not int or not 0 <= selected_class < 4:
        return None
    return selected_class


def _action_flops(action: object) -> int:
    if not isinstance(action, Mapping) or not isinstance(action.get("work"), Mapping):
        return 0
    work = action["work"]
    assert isinstance(work, Mapping)
    flops = work.get("estimated_arithmetic_flops")
    return flops if type(flops) is int and flops >= 0 else 0


def _invalid_action_flops(shot: Mapping[str, object], action_name: str) -> int:
    invalid_work = shot.get("invalid_tensor_work")
    if not isinstance(invalid_work, Mapping) or not isinstance(invalid_work.get(action_name), Mapping):
        return 0
    action_work = invalid_work[action_name]
    assert isinstance(action_work, Mapping)
    flops = action_work.get("estimated_arithmetic_flops")
    return flops if type(flops) is int and flops >= 0 else 0


def _optional_distance(
    function: object, left: np.ndarray | None, right: np.ndarray | None
) -> float | None:
    if left is None or right is None:
        return None
    return float(function(left, right))  # type: ignore[operator]


def _combined_spectral_features(
    columns: Mapping[str, object], rows: Mapping[str, object]
) -> dict[str, object]:
    available = [summary for summary in (columns, rows) if summary["available"]]
    discarded = [
        float(summary["maximum_discarded_squared_weight_fraction"])
        for summary in available
        if summary["maximum_discarded_squared_weight_fraction"] is not None
    ]
    ranks = [
        int(summary["maximum_retained_rank"])
        for summary in available
        if summary["maximum_retained_rank"] is not None
    ]
    entropy_numerators = [
        float(summary["mean_spectral_entropy"]) * int(summary["spectrum_count"])
        for summary in available
        if summary["mean_spectral_entropy"] is not None
    ]
    spectrum_count = sum(int(summary["spectrum_count"]) for summary in available)
    return {
        "tolerance_maximum_discarded_squared_weight_fraction": (
            max(discarded) if discarded else None
        ),
        "tolerance_mean_spectral_entropy": (
            sum(entropy_numerators) / spectrum_count if spectrum_count else None
        ),
        "tolerance_maximum_retained_rank": max(ranks) if ranks else None,
    }


def extract_diagnostic_row(shot: Mapping[str, object]) -> dict[str, object]:
    """Extract a compact row while separating deployable features from labels."""
    tensor = shot.get("tensor")
    arms = shot.get("arms")
    work = shot.get("work")
    certificate = shot.get("reference_certificate")
    if not all(isinstance(value, Mapping) for value in (tensor, arms, work, certificate)):
        raise ValueError("shot is missing tensor, arm, work, or reference mappings")
    assert isinstance(tensor, Mapping)
    assert isinstance(arms, Mapping)
    assert isinstance(work, Mapping)
    assert isinstance(certificate, Mapping)

    tolerance_columns_action = tensor.get("tolerance_columns")
    tolerance_rows_action = tensor.get("tolerance_rows")
    fixed_action = tensor.get("fixed_columns")
    exact_action = tensor.get("exact_columns")
    tolerance_columns = _action_probabilities(tolerance_columns_action)
    tolerance_rows = _action_probabilities(tolerance_rows_action)
    fixed = _action_probabilities(fixed_action)
    exact = _action_probabilities(exact_action)
    tolerance_columns_spectra = summarize_spectra(
        tolerance_columns_action if isinstance(tolerance_columns_action, Mapping) else None
    )
    tolerance_rows_spectra = summarize_spectra(
        tolerance_rows_action if isinstance(tolerance_rows_action, Mapping) else None
    )

    tolerance_log_ratio = (
        maximum_log_ratio_discrepancy(tolerance_columns, tolerance_rows)
        if tolerance_columns is not None and tolerance_rows is not None
        else None
    )
    columns_class = _selected_class(tolerance_columns_action)
    rows_class = _selected_class(tolerance_rows_action)
    exact_class = shot.get("exact_class")
    policy_class = shot.get("policy_class")
    if type(exact_class) is not int or type(policy_class) is not int:
        raise ValueError("shot has invalid exact or policy class")
    probability_margins = certificate.get("probability_margins")
    if not isinstance(probability_margins, list) or not probability_margins:
        raise ValueError("shot has invalid reference probability margins")
    exact_margin = min(float(value) for value in probability_margins)

    exact_arm = arms.get("exact")
    policy_arm = arms.get("policy")
    if not isinstance(exact_arm, Mapping) or not isinstance(policy_arm, Mapping):
        raise TypeError("shot is missing exact or policy arm")
    exact_failure = exact_arm.get("failure")
    policy_failure = policy_arm.get("failure")
    if type(exact_failure) is not bool or type(policy_failure) is not bool:
        raise ValueError("shot has invalid exact or policy failure outcome")
    outcome_discordance = shot.get("outcome_discordance")
    if type(outcome_discordance) is not bool:
        raise ValueError("shot has invalid outcome discordance")

    cheap_flops = (
        _action_flops(tolerance_columns_action)
        + _action_flops(tolerance_rows_action)
        + _invalid_action_flops(shot, "tolerance_columns")
        + _invalid_action_flops(shot, "tolerance_rows")
    )
    inference_features: dict[str, object] = {
        "tolerance_views_valid": tolerance_columns is not None and tolerance_rows is not None,
        "tolerance_class_agreement": (
            columns_class is not None and rows_class is not None and columns_class == rows_class
        ),
        "tolerance_total_variation": _optional_distance(
            total_variation, tolerance_columns, tolerance_rows
        ),
        "tolerance_jensen_shannon": _optional_distance(
            jensen_shannon_divergence, tolerance_columns, tolerance_rows
        ),
        "tolerance_log_ratio_discrepancy": tolerance_log_ratio,
        "tolerance_column_margin": (
            probability_margin(tolerance_columns) if tolerance_columns is not None else None
        ),
        "tolerance_row_margin": (
            probability_margin(tolerance_rows) if tolerance_rows is not None else None
        ),
        "tolerance_minimum_margin": (
            min(probability_margin(tolerance_columns), probability_margin(tolerance_rows))
            if tolerance_columns is not None and tolerance_rows is not None
            else None
        ),
        "tolerance_column_spectra": tolerance_columns_spectra,
        "tolerance_row_spectra": tolerance_rows_spectra,
        "cheap_estimated_arithmetic_flops": cheap_flops,
        **_combined_spectral_features(tolerance_columns_spectra, tolerance_rows_spectra),
    }
    return {
        "identity": {
            "shot_id": shot.get("shot_id"),
            "shot_index": shot.get("shot_index"),
            "error_rate": shot.get("error_rate"),
        },
        "decision": {
            "accepted": shot.get("accepted"),
            "used_fallback": shot.get("used_fallback"),
            "reference_valid": shot.get("reference_valid"),
        },
        "inference_features": inference_features,
        "post_refinement_diagnostics": {
            "column_to_fixed_chi8_total_variation": _optional_distance(
                total_variation, tolerance_columns, fixed
            ),
            "fixed_chi8_to_reference_total_variation": _optional_distance(
                total_variation, fixed, exact
            ),
            "fixed_chi8_class_matches_reference": (
                _selected_class(fixed_action) == exact_class
                if _selected_class(fixed_action) is not None
                else None
            ),
        },
        "reference_labels": {
            "exact_class": exact_class,
            "policy_class": policy_class,
            "class_mismatch": shot.get("class_mismatch"),
            "exact_margin": exact_margin,
            "column_to_reference_total_variation": _optional_distance(
                total_variation, tolerance_columns, exact
            ),
            "row_to_reference_total_variation": _optional_distance(
                total_variation, tolerance_rows, exact
            ),
        },
        "physical_outcome_labels": {
            "outcome_discordance": outcome_discordance,
            "exact_failure": exact_failure,
            "policy_failure": policy_failure,
            "change_helped_realized_outcome": (
                outcome_discordance and exact_failure and not policy_failure
            ),
            "change_harmed_realized_outcome": (
                outcome_discordance and policy_failure and not exact_failure
            ),
        },
        "work": {
            "cheap_estimated_arithmetic_flops": cheap_flops,
            "policy_estimated_arithmetic_flops": work.get("policy"),
            "fixed_chi8_estimated_arithmetic_flops": work.get("fixed_chi8"),
            "exact_estimated_arithmetic_flops": work.get("exact"),
        },
    }


def diagnostic_availability_ledger() -> list[dict[str, object]]:
    """Describe when each proposed signal becomes available and what it costs."""
    return [
        {
            "signal": "cross_view_disagreement",
            "earliest_stage": "cheap_action_complete",
            "fields": [
                "tolerance_total_variation",
                "tolerance_jensen_shannon",
                "tolerance_log_ratio_discrepancy",
            ],
            "cost_boundary": "requires both charged tolerance contractions",
            "deployable_from_artifact": True,
        },
        {
            "signal": "decision_margin",
            "earliest_stage": "cheap_action_complete",
            "fields": [
                "tolerance_column_margin",
                "tolerance_row_margin",
                "tolerance_minimum_margin",
            ],
            "cost_boundary": "available after each charged tolerance posterior",
            "deployable_from_artifact": True,
        },
        {
            "signal": "cross_scale_stability",
            "earliest_stage": "refinement_complete",
            "fields": ["column_to_fixed_chi8_total_variation"],
            "cost_boundary": "requires the separately charged fixed-chi8 contraction",
            "deployable_from_artifact": True,
        },
        {
            "signal": "discarded_representation",
            "earliest_stage": "cheap_action_complete",
            "fields": [
                "tolerance_column_spectra",
                "tolerance_row_spectra",
                "tolerance_maximum_discarded_squared_weight_fraction",
                "tolerance_mean_spectral_entropy",
            ],
            "cost_boundary": "observed during the charged tolerance contractions",
            "deployable_from_artifact": True,
        },
        {
            "signal": "calibrated_ood_score",
            "earliest_stage": "unavailable_without_fitted_calibrator",
            "fields": [],
            "cost_boundary": "requires a separately trained and frozen calibration model",
            "deployable_from_artifact": False,
        },
    ]
