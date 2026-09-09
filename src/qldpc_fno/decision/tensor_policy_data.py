"""Fresh, transpose-paired counterfactual tables for tensor action policies."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
from qecsim.models.generic import DepolarizingErrorModel
from qecsim.models.planar import PlanarCode

from qldpc_fno.artifacts import sha256_file, write_canonical_json
from qldpc_fno.decision.tensor_network import (
    InvalidCosetMassError,
    exact_planar_coset_masses,
    planar_mps_coset_masses,
)

_SOURCE_PATHS = (Path(__file__), Path(__file__).with_name("tensor_network.py"))
_SPLITS = ("train", "calibration", "confirmation")
_MODES = {"columns", "rows", "average"}
_FEATURE_NAMES = (
    "distance",
    "known_error_rate",
    "syndrome_density",
    "primal_density",
    "dual_density",
    "active_row_mean",
    "active_column_mean",
    "active_row_variance",
    "active_column_variance",
    "boundary_fraction",
)


def _load_config(path: Path) -> dict[str, object]:
    config = json.loads(path.read_text())
    if config.get("schema_version") != 1:
        raise ValueError("tensor-policy-data schema_version must be 1")
    domain = str(config["seed_domain"])
    expected = int.from_bytes(hashlib.sha256(domain.encode()).digest()[:8], "big")
    if int(config["campaign_seed"]) != expected:
        raise ValueError("campaign_seed must be the SHA-256 derivation of seed_domain")
    counts = config.get("base_instances_per_stratum")
    if (
        not isinstance(counts, dict)
        or set(counts) != set(_SPLITS)
        or any(type(counts[split]) is not int or counts[split] < 0 for split in _SPLITS)
        or counts["confirmation"] <= 0
    ):
        raise ValueError(
            "base_instances_per_stratum needs nonnegative counts and positive confirmation"
        )
    actions = config.get("actions")
    if not isinstance(actions, list) or not actions:
        raise ValueError("actions must be a nonempty list")
    action_ids: set[str] = set()
    action_keys: set[tuple[str, int]] = set()
    for action in actions:
        if not isinstance(action, dict):
            raise TypeError("each action must be an object")
        action_id = str(action.get("id"))
        mode = str(action.get("mode"))
        chi = action.get("chi")
        if action_id in action_ids or mode not in _MODES or type(chi) is not int or chi <= 0:
            raise ValueError("actions need unique ids, supported modes, and positive integer chi")
        if (mode, chi) in action_keys:
            raise ValueError("actions must have unique mode/chi pairs")
        action_ids.add(action_id)
        action_keys.add((mode, chi))
    for action in actions:
        counterpart = {"columns": "rows", "rows": "columns", "average": "average"}[
            str(action["mode"])
        ]
        if (counterpart, int(action["chi"])) not in action_keys:
            raise ValueError("every action requires its transpose counterpart at the same chi")
    for key in ("reference_probability_tolerance", "reference_log_ratio_tolerance"):
        if float(config.get(key, -1)) <= 0:
            raise ValueError(f"{key} must be positive")
    return config


def _seed(
    campaign_seed: int,
    *,
    split: str,
    distance: int,
    error_rate: float,
    index: int,
) -> int:
    identity = (
        f"qldpc-fno/tensor-policy/base/v1:{campaign_seed}:{split}:"
        f"{distance}:{error_rate:.17g}:{index}"
    )
    return int.from_bytes(hashlib.sha256(identity.encode()).digest()[:8], "big")


def _plaquette_indices(code: PlanarCode) -> tuple[tuple[int, int], ...]:
    primal: list[tuple[int, int]] = []
    dual: list[tuple[int, int]] = []
    max_row, max_column = code.bounds
    for index in np.ndindex((max_row + 1, max_column + 1)):
        if code.is_plaquette(index):
            (primal if code.is_primal(index) else dual).append(index)
    return tuple(primal + dual)


def _transpose_syndrome(code: PlanarCode, syndrome: np.ndarray) -> np.ndarray:
    indices = _plaquette_indices(code)
    positions = {index: position for position, index in enumerate(indices)}
    transposed = np.zeros_like(syndrome)
    for position, value in enumerate(syndrome):
        if value:
            row, column = indices[position]
            transposed[positions[(column, row)]] = value
    return transposed


def _features(code: PlanarCode, syndrome: np.ndarray, error_rate: float) -> tuple[list[float], list[list[int]]]:
    indices = _plaquette_indices(code)
    active = [index for index, value in zip(indices, syndrome) if value]
    primal_total = sum(code.is_primal(index) for index in indices)
    dual_total = len(indices) - primal_total
    primal_active = sum(code.is_primal(index) for index in active)
    dual_active = len(active) - primal_active
    max_row, max_column = code.bounds
    if active:
        rows = np.asarray([index[0] / max_row for index in active], dtype=np.float64)
        columns = np.asarray([index[1] / max_column for index in active], dtype=np.float64)
        row_mean, column_mean = float(rows.mean()), float(columns.mean())
        row_variance, column_variance = float(rows.var()), float(columns.var())
        boundary_fraction = float(
            np.mean(
                [
                    row in (0, max_row) or column in (0, max_column)
                    for row, column in active
                ]
            )
        )
    else:
        row_mean = column_mean = row_variance = column_variance = boundary_fraction = 0.0
    vector = [
        float(code.n_k_d[2]),
        error_rate,
        len(active) / len(indices),
        primal_active / primal_total,
        dual_active / dual_total,
        row_mean,
        column_mean,
        row_variance,
        column_variance,
        boundary_fraction,
    ]
    grid = np.zeros((max_row + 1, max_column + 1), dtype=np.uint8)
    for index in active:
        grid[index] = 1
    return vector, grid.tolist()


def _log_ratio_error(candidate: np.ndarray, reference: np.ndarray) -> float:
    candidate_logs = np.log(candidate)
    reference_logs = np.log(reference)
    return max(
        abs(
            float(candidate_logs[first] - candidate_logs[second])
            - float(reference_logs[first] - reference_logs[second])
        )
        for first in range(4)
        for second in range(first + 1, 4)
    )


def _context(
    *,
    config: dict[str, object],
    code: PlanarCode,
    group_id: str,
    split: str,
    orientation: str,
    syndrome: np.ndarray,
    error_rate: float,
) -> tuple[dict[str, object], float, float, float | None, float | None]:
    reference_columns = planar_mps_coset_masses(
        rows=code.size[0],
        columns=code.size[1],
        syndrome=syndrome,
        error_rate=error_rate,
        chi=None,
        mode="columns",
    )
    reference_rows = planar_mps_coset_masses(
        rows=code.size[0],
        columns=code.size[1],
        syndrome=syndrome,
        error_rate=error_rate,
        chi=None,
        mode="rows",
    )
    reference_difference = float(
        np.max(np.abs(reference_columns.probabilities - reference_rows.probabilities))
    )
    reference_log_ratio_difference = _log_ratio_error(
        reference_columns.probabilities, reference_rows.probabilities
    )
    reference_valid = (
        reference_columns.selected_class == reference_rows.selected_class
        and reference_difference <= float(config["reference_probability_tolerance"])
        and reference_log_ratio_difference <= float(config["reference_log_ratio_tolerance"])
    )
    reference_probabilities = (
        reference_columns.probabilities + reference_rows.probabilities
    ) / 2
    reference_probabilities /= reference_probabilities.sum()
    enumeration_difference: float | None = None
    enumeration_log_ratio_difference: float | None = None
    if code.n_k_d[2] == 3 and bool(config["enumerate_distance3"]):
        enumerated = exact_planar_coset_masses(
            rows=3,
            columns=3,
            syndrome=syndrome,
            error_rate=error_rate,
        )
        enumeration_difference = float(
            np.max(np.abs(reference_probabilities - enumerated.probabilities))
        )
        enumeration_log_ratio_difference = _log_ratio_error(
            reference_probabilities, enumerated.probabilities
        )
        reference_valid = (
            reference_valid
            and enumeration_difference <= float(config["reference_probability_tolerance"])
            and enumeration_log_ratio_difference
            <= float(config["reference_log_ratio_tolerance"])
        )

    features, grid = _features(code, syndrome, error_rate)
    outcomes: list[dict[str, object]] = []
    if reference_valid:
        for action in config["actions"]:
            try:
                result = planar_mps_coset_masses(
                    rows=code.size[0],
                    columns=code.size[1],
                    syndrome=syndrome,
                    error_rate=error_rate,
                    chi=int(action["chi"]),
                    mode=str(action["mode"]),
                    trace_work=True,
                )
                probabilities = result.probabilities
                selected_class = result.selected_class
                valid = True
                work = {
                    key: value
                    for key, value in result.work.items()
                    if key != "single_shot_wall_seconds"
                }
                log_error = _log_ratio_error(probabilities, reference_probabilities)
                decision_regret = float(
                    reference_probabilities.max() - reference_probabilities[selected_class]
                )
            except InvalidCosetMassError as failure:
                probabilities = None
                selected_class = None
                valid = False
                work = {
                    key: value
                    for key, value in failure.work.items()
                    if key != "single_shot_wall_seconds"
                }
                log_error = decision_regret = None
            class_correct = valid and selected_class == int(np.argmax(reference_probabilities))
            outcomes.append(
                {
                    "action_id": str(action["id"]),
                    "mode": str(action["mode"]),
                    "chi": int(action["chi"]),
                    "solver_valid": valid,
                    "selected_class": selected_class,
                    "selected_class_correct": class_correct,
                    "decision_regret": decision_regret,
                    "maximum_log_mass_ratio_error": log_error,
                    "mass_fidelity_feasible": (
                        valid
                        and class_correct
                        and float(log_error) <= float(config["mass_log_ratio_tolerance"])
                    ),
                    "probabilities": None if probabilities is None else probabilities.tolist(),
                    "work": work,
                }
            )
    return (
        {
            "context_id": f"{group_id}/{orientation}",
            "group_id": group_id,
            "split": split,
            "orientation": orientation,
            "distance": code.n_k_d[2],
            "known_error_rate": error_rate,
            "syndrome": syndrome.tolist(),
            "syndrome_grid": grid,
            "feature_names": list(_FEATURE_NAMES),
            "features": features,
            "reference_valid": reference_valid,
            "reference_probabilities": reference_probabilities.tolist(),
            "reference_selected_class": int(np.argmax(reference_probabilities)),
            "reference_row_column_probability_difference": reference_difference,
            "reference_row_column_log_ratio_difference": reference_log_ratio_difference,
            "reference_enumeration_probability_difference": enumeration_difference,
            "reference_enumeration_log_ratio_difference": enumeration_log_ratio_difference,
            "outcomes": outcomes,
        },
        reference_difference,
        reference_log_ratio_difference,
        enumeration_difference,
        enumeration_log_ratio_difference,
    )


def _transpose_audit(
    contexts: list[dict[str, object]], action_map: dict[str, str]
) -> dict[str, object]:
    class_permutation = [0, 3, 2, 1]
    maximum_probability_difference = 0.0
    validity_mismatches = 0
    feasibility_mismatches = 0
    for group_id in {str(row["group_id"]) for row in contexts}:
        original = next(
            row for row in contexts if row["group_id"] == group_id and row["orientation"] == "original"
        )
        transpose = next(
            row for row in contexts if row["group_id"] == group_id and row["orientation"] == "transpose"
        )
        if not original["reference_valid"] or not transpose["reference_valid"]:
            continue
        transpose_outcomes = {row["action_id"]: row for row in transpose["outcomes"]}
        for outcome in original["outcomes"]:
            paired = transpose_outcomes[action_map[str(outcome["action_id"])]]
            validity_mismatches += int(outcome["solver_valid"] != paired["solver_valid"])
            feasibility_mismatches += int(
                outcome["mass_fidelity_feasible"] != paired["mass_fidelity_feasible"]
            )
            if outcome["solver_valid"] and paired["solver_valid"]:
                expected = np.asarray(outcome["probabilities"])[class_permutation]
                maximum_probability_difference = max(
                    maximum_probability_difference,
                    float(np.max(np.abs(expected - np.asarray(paired["probabilities"])))),
                )
    return {
        "class_permutation_original_to_transpose": class_permutation,
        "maximum_action_probability_difference": maximum_probability_difference,
        "solver_validity_mismatch_count": validity_mismatches,
        "mass_fidelity_feasibility_mismatch_count": feasibility_mismatches,
    }


def generate_tensor_policy_data(config_path: Path, output_dir: Path) -> dict[str, object]:
    """Generate fresh grouped splits with complete paired action outcomes."""
    config = _load_config(config_path)
    model = DepolarizingErrorModel()
    base_instances: list[dict[str, object]] = []
    contexts: list[dict[str, object]] = []
    reference_differences: list[float] = []
    reference_log_ratio_differences: list[float] = []
    enumeration_differences: list[float] = []
    enumeration_log_ratio_differences: list[float] = []

    for split in _SPLITS:
        for distance in config["code_distances"]:
            code = PlanarCode(int(distance), int(distance))
            for rate_value in config["error_rates"]:
                error_rate = float(rate_value)
                for index in range(int(config["base_instances_per_stratum"][split])):
                    seed = _seed(
                        int(config["campaign_seed"]),
                        split=split,
                        distance=int(distance),
                        error_rate=error_rate,
                        index=index,
                    )
                    error = model.generate(code, error_rate, np.random.default_rng(seed))
                    syndrome = np.asarray(error @ code.stabilizers.T % 2, dtype=np.uint8)
                    group_id = f"{split}/d{distance}/p{error_rate:.6f}/i{index:04d}"
                    base_instances.append(
                        {
                            "group_id": group_id,
                            "split": split,
                            "distance": int(distance),
                            "error_rate": error_rate,
                            "instance_index": index,
                            "sampler_seed": seed,
                            "syndrome": syndrome.tolist(),
                        }
                    )
                    for orientation, oriented_syndrome in (
                        ("original", syndrome),
                        ("transpose", _transpose_syndrome(code, syndrome)),
                    ):
                        (
                            context,
                            reference_difference,
                            reference_log_ratio_difference,
                            enumeration_difference,
                            enumeration_log_ratio_difference,
                        ) = _context(
                            config=config,
                            code=code,
                            group_id=group_id,
                            split=split,
                            orientation=orientation,
                            syndrome=oriented_syndrome,
                            error_rate=error_rate,
                        )
                        contexts.append(context)
                        reference_differences.append(reference_difference)
                        reference_log_ratio_differences.append(reference_log_ratio_difference)
                        if enumeration_difference is not None:
                            enumeration_differences.append(enumeration_difference)
                        if enumeration_log_ratio_difference is not None:
                            enumeration_log_ratio_differences.append(
                                enumeration_log_ratio_difference
                            )

    actions_by_key = {
        (str(action["mode"]), int(action["chi"])): str(action["id"])
        for action in config["actions"]
    }
    transpose_action_map = {
        str(action["id"]): actions_by_key[
            (
                {"columns": "rows", "rows": "columns", "average": "average"}[
                    str(action["mode"])
                ],
                int(action["chi"]),
            )
        ]
        for action in config["actions"]
    }
    payload: dict[str, object] = {
        "schema_version": 1,
        "run_label": "counterfactual_policy_data_no_policy_claim",
        "config": config,
        "feature_boundary": {
            "feature_names": list(_FEATURE_NAMES),
            "known_error_rate_assumption": True,
            "reference_or_physical_error_in_features": False,
        },
        "transpose_action_map": transpose_action_map,
        "base_instances": base_instances,
        "contexts": contexts,
        "reference_audit": {
            "invalid_context_count": sum(not bool(row["reference_valid"]) for row in contexts),
            "maximum_row_column_probability_difference": max(reference_differences),
            "maximum_row_column_log_ratio_difference": max(
                reference_log_ratio_differences
            ),
            "maximum_enumeration_probability_difference": (
                max(enumeration_differences) if enumeration_differences else None
            ),
            "maximum_enumeration_log_ratio_difference": (
                max(enumeration_log_ratio_differences)
                if enumeration_log_ratio_differences
                else None
            ),
        },
        "transpose_audit": _transpose_audit(contexts, transpose_action_map),
        "provenance": {
            "config_path": str(config_path),
            "config_sha256": sha256_file(config_path),
            "source_sha256": {str(path): sha256_file(path) for path in _SOURCE_PATHS},
        },
    }
    write_canonical_json(output_dir / "tensor_policy_data.json", payload)
    return payload
