"""Locked class-only confirmation for two-view tolerance consensus."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from qldpc_fno.artifacts import sha256_file, write_canonical_json
from qldpc_fno.decision.tensor_consensus import (
    _apply_policy,
    _data_integrity,
    _stratified_bootstrap_savings,
    _stratified_safety,
    _transpose_consistency,
)
from qldpc_fno.decision.tensor_network import (
    InvalidCosetMassError,
    exact_planar_coset_masses,
    planar_mps_coset_masses,
)
from qldpc_fno.decision.tensor_policy_data import _log_ratio_error

_HISTORICAL_GENERATOR_SHA256 = "f6c9d5d8c08b6c8ea5074c056c3ca6b1d1b98659fc8b40661c2c1ca65c67e19c"


def _historical_freeze_source_present() -> bool:
    """Prevent a corrected rerun from reusing the revealed v1 confirmation domain."""
    return (
        sha256_file(Path(__file__).with_name("tensor_policy_data.py"))
        == _HISTORICAL_GENERATOR_SHA256
    )

_CANONICAL_POLICY: dict[str, object] = {
    "schema_version": 1,
    "policy_id": "two_view_tol001_consensus_v1",
    "direct_action_by_distance": {},
    "consensus_by_distance": {
        "3": {
            "actions": ["columns_tol001", "rows_tol001"],
            "fallback": "columns_chi8",
        },
        "5": {
            "actions": ["columns_tol001", "rows_tol001"],
            "fallback": "columns_chi8",
        },
    },
    "fixed_fallback_action": "columns_chi8",
    "required_data_seed_domain": "qldpc-fno/tensor-tolerance-consensus-confirmation/v1",
    "required_confirmation_per_stratum": 160,
    "familywise_alpha": 0.05,
    "familywise_comparisons": 12,
    "maximum_unsafe_rate": 0.05,
    "minimum_work_savings_fraction": 0.05,
    "work_savings_confidence_alpha": 0.05,
    "work_savings_bootstrap_replicates": 10000,
    "work_savings_bootstrap_seed": 13145023807022351235,
    "class_only_endpoint": True,
}

_CANONICAL_DATA_CONFIG: dict[str, object] = {
    "schema_version": 1,
    "seed_domain": "qldpc-fno/tensor-tolerance-consensus-confirmation/v1",
    "campaign_seed": 2908693450883826828,
    "code_distances": [3, 5],
    "error_rates": [0.05, 0.1, 0.15],
    "base_instances_per_stratum": {"train": 0, "calibration": 0, "confirmation": 160},
    "actions": [
        {"id": "columns_tol001", "mode": "columns", "tol": 0.01},
        {"id": "rows_tol001", "mode": "rows", "tol": 0.01},
        {"id": "columns_chi8", "mode": "columns", "chi": 8},
        {"id": "rows_chi8", "mode": "rows", "chi": 8},
    ],
    "work_trace_detail": "aggregate",
    "mass_log_ratio_tolerance": 0.05,
    "reference_probability_tolerance": 1e-10,
    "reference_log_ratio_tolerance": 1e-8,
    "enumerate_distance3": True,
}

_CANONICAL_ACTION_MAP = {
    "columns_tol001": "rows_tol001",
    "rows_tol001": "columns_tol001",
    "columns_chi8": "rows_chi8",
    "rows_chi8": "columns_chi8",
}
_SCIENTIFIC_SOURCE_PATHS = (
    Path(__file__),
    Path(__file__).with_name("tensor_consensus.py"),
    Path(__file__).with_name("tensor_policy_data.py"),
    Path(__file__).with_name("tensor_network.py"),
)


def _load_policy(path: Path) -> dict[str, object]:
    policy = json.loads(path.read_text())
    if policy.get("schema_version") != 1:
        raise ValueError("tensor-tolerance policy schema_version must be 1")
    if policy.get("class_only_endpoint") is not True:
        raise ValueError("class_only_endpoint must be true")
    if int(policy.get("familywise_comparisons", 0)) <= 0:
        raise ValueError("familywise_comparisons must be positive")
    return policy


def _action_metadata_integrity(data: dict[str, object]) -> bool:
    expected = {str(action["id"]): action for action in _CANONICAL_DATA_CONFIG["actions"]}
    for context in data.get("contexts", []):
        reference = np.asarray(context.get("reference_probabilities"), dtype=np.float64)
        if (
            reference.shape != (4,)
            or not np.all(np.isfinite(reference))
            or np.any(reference <= 0)
            or not np.isclose(reference.sum(), 1.0, rtol=0, atol=1e-12)
            or context.get("reference_selected_class") != int(np.argmax(reference))
        ):
            return False
        outcomes = context.get("outcomes", [])
        if len(outcomes) != len(expected):
            return False
        for outcome in outcomes:
            action = expected.get(str(outcome.get("action_id")))
            if action is None:
                return False
            if (
                outcome.get("mode") != action["mode"]
                or outcome.get("chi") != action.get("chi")
                or outcome.get("tol") != action.get("tol")
            ):
                return False
            work = outcome.get("work")
            if not isinstance(work, dict):
                return False
            if "truncation_events" in work or "contraction_sweeps" in work:
                return False
            counters = (
                "einsum_estimated_flops",
                "estimated_dense_decomposition_flops",
                "estimated_arithmetic_flops",
            )
            if any(type(work.get(key)) is not int or work[key] < 0 for key in counters):
                return False
            if work["estimated_arithmetic_flops"] != (
                work["einsum_estimated_flops"]
                + work["estimated_dense_decomposition_flops"]
            ):
                return False
            valid = outcome.get("solver_valid")
            if type(valid) is not bool:
                return False
            selected_class = outcome.get("selected_class")
            probabilities = outcome.get("probabilities")
            expected_correct = valid and selected_class == context["reference_selected_class"]
            if outcome.get("selected_class_correct") is not expected_correct:
                return False
            if valid:
                candidate = np.asarray(probabilities, dtype=np.float64)
                if (
                    type(selected_class) is not int
                    or selected_class not in range(4)
                    or candidate.shape != (4,)
                    or not np.all(np.isfinite(candidate))
                    or np.any(candidate <= 0)
                    or not np.isclose(candidate.sum(), 1.0, rtol=0, atol=1e-12)
                    or int(np.argmax(candidate)) != selected_class
                ):
                    return False
                expected_regret = float(reference.max() - reference[selected_class])
                if not np.isclose(
                    outcome.get("decision_regret"), expected_regret, rtol=0, atol=1e-12
                ):
                    return False
            elif selected_class is not None or probabilities is not None:
                return False
    return True


def _generation_provenance_integrity(data: dict[str, object]) -> bool:
    provenance = data.get("provenance")
    if not isinstance(provenance, dict):
        return False
    config_path = provenance.get("config_path")
    if not isinstance(config_path, str) or not Path(config_path).is_file():
        return False
    expected_sources = {
        str(path): sha256_file(path) for path in _SCIENTIFIC_SOURCE_PATHS[2:]
    }
    return (
        provenance.get("config_sha256") == sha256_file(Path(config_path))
        and provenance.get("source_sha256") == expected_sources
    )


def _outcome_replay_integrity(data: dict[str, object]) -> bool:
    """Recompute references, action masses, and work independently of stored labels."""
    actions = {str(action["id"]): action for action in data["config"]["actions"]}
    config = data["config"]
    row_column_probability_differences: list[float] = []
    row_column_log_differences: list[float] = []
    enumeration_probability_differences: list[float] = []
    enumeration_log_differences: list[float] = []
    for context in data["contexts"]:
        distance = int(context["distance"])
        syndrome = np.asarray(context["syndrome"], dtype=np.uint8)
        error_rate = float(context["known_error_rate"])
        reference_columns = planar_mps_coset_masses(
            rows=distance,
            columns=distance,
            syndrome=syndrome,
            error_rate=error_rate,
            chi=None,
            mode="columns",
        )
        reference_rows = planar_mps_coset_masses(
            rows=distance,
            columns=distance,
            syndrome=syndrome,
            error_rate=error_rate,
            chi=None,
            mode="rows",
        )
        reference = (
            reference_columns.probabilities + reference_rows.probabilities
        ) / 2
        reference /= reference.sum()
        row_column_probability_difference = float(
            np.max(np.abs(reference_columns.probabilities - reference_rows.probabilities))
        )
        row_column_log_difference = float(
            _log_ratio_error(reference_columns.probabilities, reference_rows.probabilities)
        )
        row_column_probability_differences.append(row_column_probability_difference)
        row_column_log_differences.append(row_column_log_difference)
        if (
            reference_columns.selected_class != reference_rows.selected_class
            or row_column_probability_difference
            > float(config["reference_probability_tolerance"])
            or row_column_log_difference > float(config["reference_log_ratio_tolerance"])
            or not np.isclose(
                row_column_probability_difference,
                context["reference_row_column_probability_difference"],
                rtol=0,
                atol=1e-12,
            )
            or not np.isclose(
                row_column_log_difference,
                context["reference_row_column_log_ratio_difference"],
                rtol=0,
                atol=1e-12,
            )
            or not np.allclose(
                reference,
                context["reference_probabilities"],
                rtol=0,
                atol=1e-12,
            )
            or int(np.argmax(reference)) != context["reference_selected_class"]
        ):
            return False
        if distance == 3 and bool(config["enumerate_distance3"]):
            enumerated = exact_planar_coset_masses(
                rows=3,
                columns=3,
                syndrome=syndrome,
                error_rate=error_rate,
            )
            enumeration_probability_difference = float(
                np.max(np.abs(reference - enumerated.probabilities))
            )
            enumeration_log_difference = float(
                _log_ratio_error(reference, enumerated.probabilities)
            )
            enumeration_probability_differences.append(enumeration_probability_difference)
            enumeration_log_differences.append(enumeration_log_difference)
            if (
                enumerated.selected_class != int(np.argmax(reference))
                or enumeration_probability_difference
                > float(config["reference_probability_tolerance"])
                or enumeration_log_difference
                > float(config["reference_log_ratio_tolerance"])
                or not np.isclose(
                    enumeration_probability_difference,
                    context["reference_enumeration_probability_difference"],
                    rtol=0,
                    atol=1e-12,
                )
                or not np.isclose(
                    enumeration_log_difference,
                    context["reference_enumeration_log_ratio_difference"],
                    rtol=0,
                    atol=1e-12,
                )
            ):
                return False
        elif (
            context["reference_enumeration_probability_difference"] is not None
            or context["reference_enumeration_log_ratio_difference"] is not None
        ):
            return False
        for stored in context["outcomes"]:
            action = actions[str(stored["action_id"])]
            try:
                result = planar_mps_coset_masses(
                    rows=distance,
                    columns=distance,
                    syndrome=syndrome,
                    error_rate=error_rate,
                    chi=int(action["chi"]) if "chi" in action else None,
                    mode=str(action["mode"]),
                    tol=float(action["tol"]) if "tol" in action else None,
                    trace_work=True,
                )
                replay_valid = True
                replay_class = result.selected_class
                replay_probabilities = result.probabilities
                work = result.work
            except InvalidCosetMassError as failure:
                replay_valid = False
                replay_class = None
                replay_probabilities = None
                work = failure.work
            if work is None or (
                stored["solver_valid"] != replay_valid
                or stored["selected_class"] != replay_class
            ):
                return False
            if replay_valid and not np.allclose(
                replay_probabilities, stored["probabilities"], rtol=0, atol=1e-12
            ):
                return False
            for key in (
                "einsum_estimated_flops",
                "estimated_dense_decomposition_flops",
                "estimated_arithmetic_flops",
            ):
                if stored["work"][key] != work[key]:
                    return False
    audit = data["reference_audit"]
    observed_audit = {
        "invalid_context_count": 0,
        "maximum_row_column_probability_difference": max(
            row_column_probability_differences
        ),
        "maximum_row_column_log_ratio_difference": max(row_column_log_differences),
        "maximum_enumeration_probability_difference": max(
            enumeration_probability_differences
        )
        if enumeration_probability_differences
        else None,
        "maximum_enumeration_log_ratio_difference": max(enumeration_log_differences)
        if enumeration_log_differences
        else None,
    }
    for key, observed in observed_audit.items():
        stored = audit.get(key)
        if observed is None:
            if stored is not None:
                return False
        elif not np.isclose(observed, stored, rtol=0, atol=1e-12):
            return False
    return True


def run_tensor_tolerance_consensus_study(
    policy_config_path: Path,
    data_path: Path,
    output_dir: Path,
) -> dict[str, object]:
    """Evaluate the frozen tolerance consensus on independent original draws."""
    policy = _load_policy(policy_config_path)
    data = json.loads(data_path.read_text())
    action_map = {str(key): str(value) for key, value in data["transpose_action_map"].items()}
    identity = {action_id: action_id for action_id in action_map}
    contexts = [
        row
        for row in data["contexts"]
        if row["split"] == "confirmation" and row["orientation"] == "original"
    ]
    fixed_id = str(policy["fixed_fallback_action"])
    decision_rows: list[dict[str, object]] = []
    for context in contexts:
        result = _apply_policy(context, policy, identity)
        fixed = next(
            outcome for outcome in context["outcomes"] if outcome["action_id"] == fixed_id
        )
        decision_rows.append(
            {
                "group_id": context["group_id"],
                "independent_unit": "base_group_original_orientation",
                "distance": int(context["distance"]),
                "known_error_rate": float(context["known_error_rate"]),
                **result,
                "fixed_action": fixed_id,
                "fixed_selected_class_correct": bool(fixed["selected_class_correct"]),
                "fixed_estimated_arithmetic_flops": int(
                    fixed["work"]["estimated_arithmetic_flops"]
                ),
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
    fixed_safety = _stratified_safety(
        decision_rows,
        field="fixed_selected_class_correct",
        adjusted_alpha=adjusted_alpha,
    )
    work_inference = _stratified_bootstrap_savings(
        decision_rows,
        replicates=int(policy["work_savings_bootstrap_replicates"]),
        seed=int(policy["work_savings_bootstrap_seed"]),
        alpha=float(policy["work_savings_confidence_alpha"]),
    )
    transpose = _transpose_consistency(
        [row for row in data["contexts"] if row["split"] == "confirmation"],
        policy,
        action_map,
    )
    expected_count = int(policy["required_confirmation_per_stratum"])
    canonical = (
        policy == _CANONICAL_POLICY
        and _historical_freeze_source_present()
        and _data_integrity(data, _CANONICAL_DATA_CONFIG, _CANONICAL_ACTION_MAP)
        and _action_metadata_integrity(data)
        and _generation_provenance_integrity(data)
        and _outcome_replay_integrity(data)
        and data["reference_audit"]["invalid_context_count"] == 0
        and len(policy_safety) == 6
        and all(row["independent_base_groups"] == expected_count for row in policy_safety)
        and transpose["tested_group_count"] == 960
        and transpose["selected_class_mismatch_count"] == 0
        and transpose["estimated_work_mismatch_count"] == 0
    )
    maximum_unsafe = float(policy["maximum_unsafe_rate"])
    fixed_passes = all(
        float(row["one_sided_wilson_upper"]) <= maximum_unsafe for row in fixed_safety
    )
    policy_passes = all(
        float(row["one_sided_wilson_upper"]) <= maximum_unsafe for row in policy_safety
    )
    if not canonical:
        status = "unresolved_noncanonical_data"
    elif not fixed_passes:
        status = "unresolved_fixed_fallback_failed_safety"
    elif not policy_passes:
        status = "falsified_tolerance_consensus_safety"
    elif float(work_inference["one_sided_lower_confidence_bound"]) < float(
        policy["minimum_work_savings_fraction"]
    ):
        status = "falsified_minimum_work_savings"
    else:
        status = "confirmed_class_safe_work_savings"

    policy_work = sum(int(row["estimated_arithmetic_flops"]) for row in decision_rows)
    fixed_work = sum(int(row["fixed_estimated_arithmetic_flops"]) for row in decision_rows)
    payload: dict[str, object] = {
        "schema_version": 1,
        "run_label": "locked_confirmation" if canonical else "reduced_nonconfirmatory",
        "gate5_tolerance_confirmation_status": status,
        "policy": policy,
        "posterior_mass_fidelity_claim": False,
        "statistical_contract": {
            "endpoint": "exact-reference logical-class agreement only",
            "independent_unit": "base physical-channel draw; transpose is paired augmentation only",
            "familywise_alpha": float(policy["familywise_alpha"]),
            "familywise_comparisons": int(policy["familywise_comparisons"]),
            "per_comparison_alpha": adjusted_alpha,
            "interval": "one-sided Wilson upper bound",
            "maximum_unsafe_rate": maximum_unsafe,
        },
        "decision_rows": decision_rows,
        "policy_safety_by_stratum": policy_safety,
        "fixed_safety_by_stratum": fixed_safety,
        "work": {
            "policy_estimated_arithmetic_flops": policy_work,
            "fixed_estimated_arithmetic_flops": fixed_work,
            "work_savings_fraction": (fixed_work - policy_work) / fixed_work,
            "work_savings_inference": work_inference,
            "fallback_count": sum(bool(row["fallback_used"]) for row in decision_rows),
            "latency_claim": False,
        },
        "transpose_consistency": transpose,
        "provenance": {
            "policy_config_path": str(policy_config_path),
            "policy_config_sha256": sha256_file(policy_config_path),
            "data_path": str(data_path),
            "data_sha256": sha256_file(data_path),
            "source_sha256": {
                str(path): sha256_file(path) for path in _SCIENTIFIC_SOURCE_PATHS
            },
        },
    }
    output_dir.mkdir(parents=True, exist_ok=False)
    write_canonical_json(output_dir / "tensor_tolerance_consensus.json", payload)
    return payload
