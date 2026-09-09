"""Frozen two-view consensus policy and stratified safety evaluation."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from qecsim.models.generic import DepolarizingErrorModel
from qecsim.models.planar import PlanarCode
from scipy.stats import norm

from qldpc_fno.artifacts import sha256_file, write_canonical_json
from qldpc_fno.decision.tensor_policy_data import (
    _features,
    _seed,
    _transpose_syndrome,
)

_CANONICAL_POLICY: dict[str, object] = {
    "schema_version": 1,
    "policy_id": "two_view_chi4_consensus_v1",
    "direct_action_by_distance": {"3": "columns_chi4"},
    "consensus_by_distance": {
        "5": {
            "actions": ["columns_chi4", "rows_chi4"],
            "fallback": "columns_chi8",
        }
    },
    "fixed_fallback_action": "columns_chi8",
    "required_data_seed_domain": "qldpc-fno/tensor-consensus-confirmation/v1",
    "required_confirmation_per_stratum": 160,
    "familywise_alpha": 0.05,
    "familywise_comparisons": 12,
    "maximum_unsafe_rate": 0.05,
    "minimum_work_savings_fraction": 0.05,
    "work_savings_confidence_alpha": 0.05,
    "work_savings_bootstrap_replicates": 10000,
    "work_savings_bootstrap_seed": 17608167868160228024,
}

_CANONICAL_DATA_CONFIG: dict[str, object] = {
    "schema_version": 1,
    "seed_domain": "qldpc-fno/tensor-consensus-confirmation/v1",
    "campaign_seed": 1369863511152507832,
    "code_distances": [3, 5],
    "error_rates": [0.05, 0.1, 0.15],
    "base_instances_per_stratum": {"train": 0, "calibration": 0, "confirmation": 160},
    "actions": [
        {"id": "columns_chi4", "mode": "columns", "chi": 4},
        {"id": "rows_chi4", "mode": "rows", "chi": 4},
        {"id": "columns_chi8", "mode": "columns", "chi": 8},
        {"id": "rows_chi8", "mode": "rows", "chi": 8},
    ],
    "mass_log_ratio_tolerance": 0.05,
    "reference_probability_tolerance": 1e-10,
    "reference_log_ratio_tolerance": 1e-8,
    "enumerate_distance3": True,
}
_CANONICAL_TRANSPOSE_ACTION_MAP = {
    "columns_chi4": "rows_chi4",
    "rows_chi4": "columns_chi4",
    "columns_chi8": "rows_chi8",
    "rows_chi8": "columns_chi8",
}


def wilson_upper(failures: int, trials: int, *, alpha: float) -> float:
    """Return a one-sided Wilson upper confidence bound."""
    if not 0 <= failures <= trials or trials <= 0:
        raise ValueError("Wilson counts require 0 <= failures <= positive trials")
    if not 0 < alpha < 1:
        raise ValueError("alpha must lie in (0, 1)")
    estimate = failures / trials
    z = float(norm.ppf(1 - alpha))
    z2 = z * z
    denominator = 1 + z2 / trials
    center = estimate + z2 / (2 * trials)
    radius = z * np.sqrt(estimate * (1 - estimate) / trials + z2 / (4 * trials**2))
    return float((center + radius) / denominator)


def _load_policy(path: Path) -> dict[str, object]:
    policy = json.loads(path.read_text())
    if policy.get("schema_version") != 1:
        raise ValueError("tensor-consensus policy schema_version must be 1")
    if int(policy.get("familywise_comparisons", 0)) <= 0:
        raise ValueError("familywise_comparisons must be positive")
    if not 0 < float(policy.get("familywise_alpha", 0)) < 1:
        raise ValueError("familywise_alpha must lie in (0, 1)")
    return policy


def _outcomes(context: dict[str, object]) -> dict[str, dict[str, object]]:
    return {str(row["action_id"]): row for row in context["outcomes"]}


def _work(outcome: dict[str, object]) -> int:
    return int(outcome["work"]["estimated_arithmetic_flops"])


def _apply_policy(
    context: dict[str, object],
    policy: dict[str, object],
    action_map: dict[str, str],
) -> dict[str, object]:
    outcomes = _outcomes(context)
    distance_key = str(context["distance"])
    direct = policy["direct_action_by_distance"]
    consensus = policy["consensus_by_distance"]
    fallback_used = False
    executed: list[str] = []

    if distance_key in direct:
        action_id = action_map[str(direct[distance_key])]
        selected = outcomes[action_id]
        executed.append(action_id)
    elif distance_key in consensus:
        rule = consensus[distance_key]
        action_ids = [action_map[str(action_id)] for action_id in rule["actions"]]
        candidates = [outcomes[action_id] for action_id in action_ids]
        executed.extend(action_ids)
        agree = (
            all(bool(candidate["solver_valid"]) for candidate in candidates)
            and len({candidate["selected_class"] for candidate in candidates}) == 1
        )
        if agree:
            selected = candidates[0]
        else:
            fallback_id = action_map[str(rule["fallback"])]
            selected = outcomes[fallback_id]
            executed.append(fallback_id)
            fallback_used = True
    else:
        raise ValueError(f"policy has no rule for distance {distance_key}")

    return {
        "selected_class": selected["selected_class"],
        "selected_solver_valid": bool(selected["solver_valid"]),
        "selected_class_correct": bool(selected["selected_class_correct"]),
        "decision_regret": selected["decision_regret"],
        "mass_fidelity_feasible": bool(selected["mass_fidelity_feasible"]),
        "fallback_used": fallback_used,
        "executed_actions": executed,
        "estimated_arithmetic_flops": sum(_work(outcomes[action_id]) for action_id in executed),
    }


def _stratified_safety(
    decision_rows: list[dict[str, object]],
    *,
    field: str,
    adjusted_alpha: float,
) -> list[dict[str, object]]:
    strata = sorted({(int(row["distance"]), float(row["known_error_rate"])) for row in decision_rows})
    summaries: list[dict[str, object]] = []
    for distance, error_rate in strata:
        selected = [
            row
            for row in decision_rows
            if row["distance"] == distance and row["known_error_rate"] == error_rate
        ]
        failures = sum(not bool(row[field]) for row in selected)
        summaries.append(
            {
                "distance": distance,
                "error_rate": error_rate,
                "independent_base_groups": len(selected),
                "failures": failures,
                "observed_unsafe_rate": failures / len(selected),
                "one_sided_wilson_upper": wilson_upper(
                    failures, len(selected), alpha=adjusted_alpha
                ),
            }
        )
    return summaries


def _transpose_consistency(
    contexts: list[dict[str, object]], policy: dict[str, object], action_map: dict[str, str]
) -> dict[str, object]:
    identity = {action_id: action_id for action_id in action_map}
    class_map = [0, 3, 2, 1]
    tested = 0
    selected_class_mismatches = 0
    work_mismatches = 0
    for group_id in sorted({str(row["group_id"]) for row in contexts}):
        pair = [row for row in contexts if row["group_id"] == group_id]
        if len(pair) != 2 or not all(bool(row["reference_valid"]) for row in pair):
            continue
        original = next(row for row in pair if row["orientation"] == "original")
        transpose = next(row for row in pair if row["orientation"] == "transpose")
        original_result = _apply_policy(original, policy, identity)
        transpose_result = _apply_policy(transpose, policy, action_map)
        tested += 1
        if original_result["selected_class"] is None or transpose_result["selected_class"] is None:
            selected_class_mismatches += 1
        else:
            selected_class_mismatches += int(
                transpose_result["selected_class"]
                != class_map[int(original_result["selected_class"])]
            )
        work_mismatches += int(
            transpose_result["estimated_arithmetic_flops"]
            != original_result["estimated_arithmetic_flops"]
        )
    return {
        "tested_group_count": tested,
        "selected_class_mismatch_count": selected_class_mismatches,
        "estimated_work_mismatch_count": work_mismatches,
    }


def _data_integrity(
    data: dict[str, object],
    expected_config: dict[str, object],
    expected_action_map: dict[str, str],
) -> bool:
    if data.get("config") != expected_config or data.get("transpose_action_map") != expected_action_map:
        return False
    base_instances = data.get("base_instances")
    contexts = data.get("contexts")
    if not isinstance(base_instances, list) or not isinstance(contexts, list):
        return False
    base_by_group = {str(row.get("group_id")): row for row in base_instances}
    contexts_by_group: dict[str, list[dict[str, object]]] = {}
    for context in contexts:
        contexts_by_group.setdefault(str(context.get("group_id")), []).append(context)
    if len(base_by_group) != len(base_instances) or len(
        {str(row.get("context_id")) for row in contexts}
    ) != len(contexts):
        return False
    expected_action_ids = {str(action["id"]) for action in expected_config["actions"]}
    expected_groups: set[str] = set()
    model = DepolarizingErrorModel()

    for split, count in expected_config["base_instances_per_stratum"].items():
        for distance in expected_config["code_distances"]:
            code = PlanarCode(int(distance), int(distance))
            for rate_value in expected_config["error_rates"]:
                error_rate = float(rate_value)
                for index in range(int(count)):
                    group_id = f"{split}/d{distance}/p{error_rate:.6f}/i{index:04d}"
                    expected_groups.add(group_id)
                    base = base_by_group.get(group_id)
                    if base is None:
                        return False
                    seed = _seed(
                        int(expected_config["campaign_seed"]),
                        split=str(split),
                        distance=int(distance),
                        error_rate=error_rate,
                        index=index,
                    )
                    error = model.generate(code, error_rate, np.random.default_rng(seed))
                    syndrome = np.asarray(error @ code.stabilizers.T % 2, dtype=np.uint8)
                    if base != {
                        "group_id": group_id,
                        "split": split,
                        "distance": int(distance),
                        "error_rate": error_rate,
                        "instance_index": index,
                        "sampler_seed": seed,
                        "syndrome": syndrome.tolist(),
                    }:
                        return False
                    pair = contexts_by_group.get(group_id, [])
                    if len(pair) != 2 or {row.get("orientation") for row in pair} != {
                        "original",
                        "transpose",
                    }:
                        return False
                    for orientation, expected_syndrome in (
                        ("original", syndrome),
                        ("transpose", _transpose_syndrome(code, syndrome)),
                    ):
                        context = next(row for row in pair if row["orientation"] == orientation)
                        features, grid = _features(code, expected_syndrome, error_rate)
                        if (
                            context.get("context_id") != f"{group_id}/{orientation}"
                            or context.get("split") != split
                            or int(context.get("distance", -1)) != int(distance)
                            or float(context.get("known_error_rate", -1)) != error_rate
                            or context.get("syndrome") != expected_syndrome.tolist()
                            or context.get("features") != features
                            or context.get("syndrome_grid") != grid
                            or not bool(context.get("reference_valid"))
                        ):
                            return False
                        outcomes = context.get("outcomes")
                        if not isinstance(outcomes, list) or len(outcomes) != len(expected_action_ids):
                            return False
                        if {str(row.get("action_id")) for row in outcomes} != expected_action_ids:
                            return False
    return set(base_by_group) == expected_groups and set(contexts_by_group) == expected_groups


def _stratified_bootstrap_savings(
    decision_rows: list[dict[str, object]],
    *,
    replicates: int,
    seed: int,
    alpha: float,
) -> dict[str, object]:
    if replicates <= 0 or not 0 < alpha < 1:
        raise ValueError("bootstrap requires positive replicates and alpha in (0, 1)")
    strata = sorted({(int(row["distance"]), float(row["known_error_rate"])) for row in decision_rows})
    grouped = [
        [
            row
            for row in decision_rows
            if row["distance"] == distance and row["known_error_rate"] == error_rate
        ]
        for distance, error_rate in strata
    ]
    rng = np.random.default_rng(seed)
    samples = np.empty(replicates, dtype=np.float64)
    for replicate in range(replicates):
        policy_work = 0
        fixed_work = 0
        for rows in grouped:
            indices = rng.integers(0, len(rows), size=len(rows))
            policy_work += sum(int(rows[index]["estimated_arithmetic_flops"]) for index in indices)
            fixed_work += sum(
                int(rows[index]["fixed_estimated_arithmetic_flops"]) for index in indices
            )
        samples[replicate] = (fixed_work - policy_work) / fixed_work
    observed_policy = sum(int(row["estimated_arithmetic_flops"]) for row in decision_rows)
    observed_fixed = sum(int(row["fixed_estimated_arithmetic_flops"]) for row in decision_rows)
    return {
        "estimand": "equal-stratum mixture mean work reduction relative to fixed fallback",
        "observed_savings_fraction": (observed_fixed - observed_policy) / observed_fixed,
        "one_sided_lower_confidence_bound": float(np.quantile(samples, alpha, method="lower")),
        "confidence_level": 1 - alpha,
        "bootstrap_replicates": replicates,
        "bootstrap_seed": seed,
        "resampling_unit": "base group, independently within each distance/error-rate stratum",
    }


def run_tensor_consensus_study(
    policy_config_path: Path,
    data_path: Path,
    output_dir: Path,
) -> dict[str, object]:
    """Evaluate the frozen symbolic consensus rule on original base groups."""
    policy = _load_policy(policy_config_path)
    data = json.loads(data_path.read_text())
    action_map = {str(key): str(value) for key, value in data["transpose_action_map"].items()}
    identity = {action_id: action_id for action_id in action_map}
    confirmation_contexts = [
        row
        for row in data["contexts"]
        if row["split"] == "confirmation" and row["orientation"] == "original"
    ]
    if any(not bool(row["reference_valid"]) for row in confirmation_contexts):
        raise ValueError("confirmation contains an invalid reference context")

    decision_rows: list[dict[str, object]] = []
    fixed_id = str(policy["fixed_fallback_action"])
    for context in confirmation_contexts:
        result = _apply_policy(context, policy, identity)
        fixed = _outcomes(context)[fixed_id]
        decision_rows.append(
            {
                "group_id": context["group_id"],
                "independent_unit": "base_group_original_orientation",
                "distance": int(context["distance"]),
                "known_error_rate": float(context["known_error_rate"]),
                **result,
                "fixed_action": fixed_id,
                "fixed_selected_class_correct": bool(fixed["selected_class_correct"]),
                "fixed_estimated_arithmetic_flops": _work(fixed),
            }
        )

    adjusted_alpha = float(policy["familywise_alpha"]) / int(
        policy["familywise_comparisons"]
    )
    policy_safety = _stratified_safety(
        decision_rows,
        field="selected_class_correct",
        adjusted_alpha=adjusted_alpha,
    )
    fallback_safety = _stratified_safety(
        decision_rows,
        field="fixed_selected_class_correct",
        adjusted_alpha=adjusted_alpha,
    )
    policy_work = sum(int(row["estimated_arithmetic_flops"]) for row in decision_rows)
    fixed_work = sum(int(row["fixed_estimated_arithmetic_flops"]) for row in decision_rows)
    savings_fraction = (fixed_work - policy_work) / fixed_work
    work_inference = _stratified_bootstrap_savings(
        decision_rows,
        replicates=int(policy["work_savings_bootstrap_replicates"]),
        seed=int(policy["work_savings_bootstrap_seed"]),
        alpha=float(policy["work_savings_confidence_alpha"]),
    )

    counts = data["config"]["base_instances_per_stratum"]
    expected_count = int(policy["required_confirmation_per_stratum"])
    canonical_data = (
        policy == _CANONICAL_POLICY
        and _data_integrity(data, _CANONICAL_DATA_CONFIG, _CANONICAL_TRANSPOSE_ACTION_MAP)
        and counts == {"train": 0, "calibration": 0, "confirmation": expected_count}
        and all(row["independent_base_groups"] == expected_count for row in policy_safety)
        and len(policy_safety) == 6
    )
    maximum_unsafe = float(policy["maximum_unsafe_rate"])
    fallback_passes = all(
        float(row["one_sided_wilson_upper"]) <= maximum_unsafe for row in fallback_safety
    )
    policy_passes = all(
        float(row["one_sided_wilson_upper"]) <= maximum_unsafe for row in policy_safety
    )
    if not canonical_data:
        status = "unresolved_noncanonical_data"
    elif not fallback_passes:
        status = "unresolved_fixed_fallback_failed_safety"
    elif not policy_passes:
        status = "falsified_consensus_safety"
    elif float(work_inference["one_sided_lower_confidence_bound"]) < float(
        policy["minimum_work_savings_fraction"]
    ):
        status = "falsified_minimum_work_savings"
    else:
        status = "confirmed_safe_work_savings"

    payload: dict[str, object] = {
        "schema_version": 1,
        "run_label": "locked_confirmation" if canonical_data else "reduced_nonconfirmatory",
        "gate5_confirmation_status": status,
        "policy": policy,
        "statistical_contract": {
            "independent_unit": "base physical-channel draw; transpose is paired augmentation only",
            "familywise_alpha": float(policy["familywise_alpha"]),
            "familywise_comparisons": int(policy["familywise_comparisons"]),
            "per_comparison_alpha": adjusted_alpha,
            "interval": "one-sided Wilson upper bound",
            "maximum_unsafe_rate": maximum_unsafe,
        },
        "decision_rows": decision_rows,
        "policy_safety_by_stratum": policy_safety,
        "fallback_safety_by_stratum": fallback_safety,
        "work": {
            "policy_estimated_arithmetic_flops": policy_work,
            "fixed_estimated_arithmetic_flops": fixed_work,
            "work_savings_fraction": savings_fraction,
            "work_savings_inference": work_inference,
            "fallback_count": sum(bool(row["fallback_used"]) for row in decision_rows),
            "latency_claim": False,
        },
        "transpose_consistency": _transpose_consistency(
            [row for row in data["contexts"] if row["split"] == "confirmation"],
            policy,
            action_map,
        ),
        "provenance": {
            "policy_config_path": str(policy_config_path),
            "policy_config_sha256": sha256_file(policy_config_path),
            "data_path": str(data_path),
            "data_sha256": sha256_file(data_path),
            "source_sha256": sha256_file(Path(__file__)),
        },
    }
    write_canonical_json(output_dir / "tensor_consensus.json", payload)
    return payload
