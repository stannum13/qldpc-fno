"""Development diagnostics for adaptive planar tensor inference."""

from __future__ import annotations

import re
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


def rank_auc(scores: Sequence[float], labels: Sequence[bool]) -> float:
    """Return tie-aware ROC AUC using the Mann-Whitney rank identity."""
    score_array = np.asarray(scores, dtype=np.float64)
    label_array = np.asarray(labels)
    if score_array.ndim != 1 or label_array.ndim != 1 or score_array.size != label_array.size:
        raise ValueError("scores and labels must be equal-length vectors")
    if score_array.size == 0 or np.any(~np.isfinite(score_array)):
        raise ValueError("scores must be a nonempty finite vector")
    if label_array.dtype != np.bool_:
        if np.any((label_array != 0) & (label_array != 1)):
            raise ValueError("labels must be boolean")
        label_array = label_array.astype(bool)
    positives = int(label_array.sum())
    negatives = int(label_array.size - positives)
    if positives == 0 or negatives == 0:
        raise ValueError("labels must contain both classes")

    order = np.argsort(score_array, kind="stable")
    sorted_scores = score_array[order]
    ranks = np.empty(score_array.size, dtype=np.float64)
    start = 0
    while start < sorted_scores.size:
        stop = start + 1
        while stop < sorted_scores.size and sorted_scores[stop] == sorted_scores[start]:
            stop += 1
        ranks[order[start:stop]] = 0.5 * ((start + 1) + stop)
        start = stop
    positive_rank_sum = float(ranks[label_array].sum())
    mann_whitney = positive_rank_sum - positives * (positives + 1) / 2
    return float(mann_whitney / (positives * negatives))


_MATCHED_DIAGNOSTIC_FIELDS = (
    "tolerance_total_variation",
    "tolerance_jensen_shannon",
    "tolerance_log_ratio_discrepancy",
    "tolerance_maximum_discarded_squared_weight_fraction",
    "tolerance_mean_spectral_entropy",
)


def _nested_mapping(row: Mapping[str, object], name: str) -> Mapping[str, object]:
    value = row.get(name)
    if not isinstance(value, Mapping):
        raise TypeError(f"diagnostic row is missing {name}")
    return value


def select_margin_matched_controls(
    rows: Sequence[Mapping[str, object]],
) -> list[dict[str, object]]:
    """Match each accepted mismatch to a unique same-rate benign margin neighbour."""
    mismatch_rows = sorted(
        (
            row
            for row in rows
            if _nested_mapping(row, "decision").get("accepted") is True
            and _nested_mapping(row, "reference_labels").get("class_mismatch") is True
        ),
        key=lambda row: str(_nested_mapping(row, "identity").get("shot_id")),
    )
    candidate_rows = [
        row
        for row in rows
        if _nested_mapping(row, "decision").get("accepted") is True
        and _nested_mapping(row, "reference_labels").get("class_mismatch") is False
    ]
    used: set[str] = set()
    pairs: list[dict[str, object]] = []
    for mismatch in mismatch_rows:
        mismatch_identity = _nested_mapping(mismatch, "identity")
        mismatch_features = _nested_mapping(mismatch, "inference_features")
        rate = mismatch_identity.get("error_rate")
        margin = mismatch_features.get("tolerance_minimum_margin")
        if not isinstance(margin, (int, float)) or isinstance(margin, bool):
            raise TypeError("accepted mismatch lacks a numeric cheap margin")
        eligible = []
        for control in candidate_rows:
            identity = _nested_mapping(control, "identity")
            features = _nested_mapping(control, "inference_features")
            control_id = str(identity.get("shot_id"))
            control_margin = features.get("tolerance_minimum_margin")
            if (
                identity.get("error_rate") == rate
                and control_id not in used
                and isinstance(control_margin, (int, float))
                and not isinstance(control_margin, bool)
            ):
                distance = abs(float(control_margin) - float(margin))
                eligible.append((round(distance, 15), control_id, distance, control))
        if not eligible:
            raise ValueError("no unique same-rate accepted control is available")
        _, control_id, raw_distance, control = min(eligible, key=lambda item: (item[0], item[1]))
        used.add(control_id)
        control_features = _nested_mapping(control, "inference_features")
        paired_differences: dict[str, float | None] = {}
        for field in _MATCHED_DIAGNOSTIC_FIELDS:
            mismatch_value = mismatch_features.get(field)
            control_value = control_features.get(field)
            paired_differences[field] = (
                float(mismatch_value) - float(control_value)
                if isinstance(mismatch_value, (int, float))
                and not isinstance(mismatch_value, bool)
                and isinstance(control_value, (int, float))
                and not isinstance(control_value, bool)
                else None
            )
        pairs.append(
            {
                "mismatch_shot_id": mismatch_identity.get("shot_id"),
                "control_shot_id": control_id,
                "error_rate": rate,
                "mismatch_margin": float(margin),
                "control_margin": float(control_features["tolerance_minimum_margin"]),
                "absolute_margin_difference": raw_distance,
                "paired_differences": paired_differences,
            }
        )
    return pairs


_AUDIT_SCORES = (
    ("cross_view_total_variation", "inference_features", "tolerance_total_variation", 1.0),
    ("cross_view_jensen_shannon", "inference_features", "tolerance_jensen_shannon", 1.0),
    (
        "cross_view_log_ratio_discrepancy",
        "inference_features",
        "tolerance_log_ratio_discrepancy",
        1.0,
    ),
    ("negative_minimum_margin", "inference_features", "tolerance_minimum_margin", -1.0),
    (
        "maximum_discarded_squared_weight_fraction",
        "inference_features",
        "tolerance_maximum_discarded_squared_weight_fraction",
        1.0,
    ),
    ("mean_spectral_entropy", "inference_features", "tolerance_mean_spectral_entropy", 1.0),
    (
        "post_refinement_column_drift",
        "post_refinement_diagnostics",
        "column_to_fixed_chi8_total_variation",
        1.0,
    ),
)


def _score_audit(
    rows: Sequence[Mapping[str, object]], section: str, field: str, multiplier: float
) -> dict[str, object]:
    scores: list[float] = []
    labels: list[bool] = []
    for row in rows:
        value = _nested_mapping(row, section).get(field)
        label = _nested_mapping(row, "reference_labels").get("class_mismatch")
        if (
            isinstance(value, (int, float))
            and not isinstance(value, bool)
            and np.isfinite(value)
            and type(label) is bool
        ):
            scores.append(multiplier * float(value))
            labels.append(label)
    positives = sum(labels)
    negatives = len(labels) - positives
    return {
        "source_section": section,
        "source_field": field,
        "higher_score_means": "more_suspicious",
        "observations": len(labels),
        "reference_mismatches": positives,
        "reference_matches": negatives,
        "auc": rank_auc(scores, labels) if positives and negatives else None,
        "unavailable_reason": None if positives and negatives else "requires both label classes",
    }


def _require_sha256(value: str) -> None:
    if re.fullmatch(r"[0-9a-f]{64}", value) is None:
        raise ValueError("source_sha256 must be a lowercase SHA-256 digest")


def _development_margin_gate(rows: Sequence[Mapping[str, object]]) -> dict[str, object]:
    accepted_agreements = [
        row
        for row in rows
        if _nested_mapping(row, "decision")["accepted"] is True
        and _nested_mapping(row, "inference_features")["tolerance_class_agreement"] is True
    ]
    mismatches = [
        row
        for row in accepted_agreements
        if _nested_mapping(row, "reference_labels")["class_mismatch"] is True
    ]
    mismatch_margins = [
        float(_nested_mapping(row, "inference_features")["tolerance_minimum_margin"])
        for row in mismatches
    ]
    if not mismatch_margins:
        return {
            "claim_status": "unavailable_without_revealed_reference_mismatches",
            "selected_on_revealed_labels": True,
            "threshold": None,
        }
    threshold = max(mismatch_margins)
    cheaply_accepted = [
        row
        for row in accepted_agreements
        if float(_nested_mapping(row, "inference_features")["tolerance_minimum_margin"])
        > threshold
    ]
    cheaply_accepted_ids = {
        str(_nested_mapping(row, "identity")["shot_id"]) for row in cheaply_accepted
    }
    escalated = [
        row
        for row in rows
        if str(_nested_mapping(row, "identity")["shot_id"]) not in cheaply_accepted_ids
    ]
    accepted_mismatches = sum(
        _nested_mapping(row, "reference_labels")["class_mismatch"] is True
        for row in cheaply_accepted
    )
    fallback_mismatches = sum(
        _nested_mapping(row, "post_refinement_diagnostics")[
            "fixed_chi8_class_matches_reference"
        ]
        is False
        for row in escalated
    )
    unresolved_escalations = sum(
        _nested_mapping(row, "post_refinement_diagnostics")[
            "fixed_chi8_class_matches_reference"
        ]
        is None
        for row in escalated
    )
    projected_work = sum(
        int(_nested_mapping(row, "work")["cheap_estimated_arithmetic_flops"])
        + (
            0
            if str(_nested_mapping(row, "identity")["shot_id"]) in cheaply_accepted_ids
            else int(_nested_mapping(row, "work")["fixed_chi8_estimated_arithmetic_flops"])
        )
        for row in rows
    )
    fixed_work = sum(
        int(_nested_mapping(row, "work")["fixed_chi8_estimated_arithmetic_flops"])
        for row in rows
    )
    rates = sorted({float(_nested_mapping(row, "identity")["error_rate"]) for row in rows})
    by_rate = []
    for rate in rates:
        rate_rows = [row for row in rows if float(_nested_mapping(row, "identity")["error_rate"]) == rate]
        rate_accepts = [
            row
            for row in rate_rows
            if str(_nested_mapping(row, "identity")["shot_id"]) in cheaply_accepted_ids
        ]
        by_rate.append(
            {
                "error_rate": rate,
                "shots": len(rate_rows),
                "cheap_accepts": len(rate_accepts),
                "cheap_acceptance_fraction": len(rate_accepts) / len(rate_rows),
                "accepted_reference_mismatches": sum(
                    _nested_mapping(row, "reference_labels")["class_mismatch"] is True
                    for row in rate_accepts
                ),
            }
        )
    return {
        "claim_status": "development_hypothesis_only",
        "selected_on_revealed_labels": True,
        "threshold_rule": "accept only when tolerance_minimum_margin > threshold",
        "threshold": threshold,
        "shots": len(rows),
        "cheap_accepts": len(cheaply_accepted),
        "cheap_acceptance_fraction": len(cheaply_accepted) / len(rows),
        "escalations": len(escalated),
        "accepted_reference_mismatches": accepted_mismatches,
        "fallback_reference_mismatches": fallback_mismatches,
        "unresolved_escalations": unresolved_escalations,
        "projected_final_reference_mismatches": accepted_mismatches + fallback_mismatches,
        "by_error_rate": by_rate,
        "projected_work": {
            "estimand": "cheap tolerance work on every shot plus fixed-chi8 work on escalations",
            "estimated_arithmetic_flops": projected_work,
            "fixed_chi8_comparator_flops": fixed_work,
            "ratio_to_fixed_chi8": projected_work / fixed_work,
            "saving_relative_to_fixed_chi8": 1 - projected_work / fixed_work,
        },
    }


def build_diagnostic_audit(
    source: Mapping[str, object], *, source_sha256: str
) -> dict[str, object]:
    """Build a development-only diagnostic audit from a frozen planar result."""
    _require_sha256(source_sha256)
    source_shots = source.get("shots")
    source_per_rate = source.get("per_rate")
    source_work = source.get("work")
    if (
        not isinstance(source_shots, list)
        or not isinstance(source_per_rate, list)
        or not isinstance(source_work, Mapping)
        or not isinstance(source_work.get("totals"), Mapping)
    ):
        raise TypeError("source is missing shots, per-rate aggregates, or work totals")
    rows = [extract_diagnostic_row(shot) for shot in source_shots]

    derived_per_rate: list[dict[str, object]] = []
    consistency_checks: list[dict[str, object]] = []
    for declared in source_per_rate:
        if not isinstance(declared, Mapping):
            raise TypeError("source per-rate aggregate must be a mapping")
        rate = declared.get("error_rate")
        rate_rows = [row for row in rows if _nested_mapping(row, "identity")["error_rate"] == rate]
        derived = {
            "error_rate": rate,
            "shots": len(rate_rows),
            "class_mismatches": sum(
                _nested_mapping(row, "reference_labels")["class_mismatch"] is True
                for row in rate_rows
            ),
            "outcome_discordances": sum(
                _nested_mapping(row, "physical_outcome_labels")["outcome_discordance"] is True
                for row in rate_rows
            ),
            "fallbacks": sum(
                _nested_mapping(row, "decision")["used_fallback"] is True for row in rate_rows
            ),
        }
        for field in ("shots", "class_mismatches", "outcome_discordances", "fallbacks"):
            if derived[field] != declared.get(field):
                raise ValueError(
                    f"source aggregate mismatch at error_rate={rate} for {field}: "
                    f"derived {derived[field]}, declared {declared.get(field)}"
                )
        derived_per_rate.append(derived)
        consistency_checks.append({"error_rate": rate, "checks": derived, "passed": True})

    declared_totals = source_work["totals"]
    assert isinstance(declared_totals, Mapping)
    derived_work = {
        "policy": sum(
            int(_nested_mapping(row, "work")["policy_estimated_arithmetic_flops"])
            for row in rows
        ),
        "fixed_chi8": sum(
            int(_nested_mapping(row, "work")["fixed_chi8_estimated_arithmetic_flops"])
            for row in rows
        ),
        "exact": sum(
            int(_nested_mapping(row, "work")["exact_estimated_arithmetic_flops"])
            for row in rows
        ),
    }
    for field, value in derived_work.items():
        if value != declared_totals.get(field):
            raise ValueError(
                f"source work total mismatch for {field}: derived {value}, "
                f"declared {declared_totals.get(field)}"
            )

    accepted_agreements = [
        row
        for row in rows
        if _nested_mapping(row, "decision")["accepted"] is True
        and _nested_mapping(row, "inference_features")["tolerance_class_agreement"] is True
    ]
    score_audits: dict[str, object] = {}
    for name, section, field, multiplier in _AUDIT_SCORES:
        pooled = _score_audit(accepted_agreements, section, field, multiplier)
        by_rate = {}
        for rate in sorted(
            {float(_nested_mapping(row, "identity")["error_rate"]) for row in accepted_agreements}
        ):
            rate_rows = [
                row
                for row in accepted_agreements
                if float(_nested_mapping(row, "identity")["error_rate"]) == rate
            ]
            by_rate[f"{rate:.6f}"] = _score_audit(rate_rows, section, field, multiplier)
        score_audits[name] = {"pooled": pooled, "by_error_rate": by_rate}

    mismatches = [
        row
        for row in accepted_agreements
        if _nested_mapping(row, "reference_labels")["class_mismatch"] is True
    ]
    return {
        "schema_version": 1,
        "artifact_kind": "adaptive_computation_diagnostic_audit",
        "status": "development_open_exploratory_audit",
        "source": {
            "sha256": source_sha256,
            "schema_version": source.get("schema_version"),
            "status": source.get("status"),
            "shots": len(rows),
        },
        "source_consistency": {
            "passed": True,
            "per_rate": consistency_checks,
            "work_totals": derived_work,
        },
        "availability_ledger": diagnostic_availability_ledger(),
        "rows": rows,
        "common_mode_reference_mismatches": mismatches,
        "margin_matched_controls": select_margin_matched_controls(rows),
        "development_zero_mismatch_margin_gate": _development_margin_gate(rows),
        "exploratory_discrimination": {
            "development_only": True,
            "accepted_agreement_shots": len(accepted_agreements),
            "reference_mismatches": len(mismatches),
            "scores": score_audits,
        },
        "derived_per_rate": derived_per_rate,
        "nonclaims": [
            "No threshold or diagnostic was evaluated on an untouched confirmation domain.",
            "The eight revealed reference mismatches cannot establish calibrated safety.",
            "AUCs on these development-open outcomes do not establish out-of-sample prediction.",
            "Estimated arithmetic work is not measured latency, throughput, energy, or hardware fit.",
        ],
    }
