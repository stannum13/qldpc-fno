"""Discovery study for local-spectrum tolerance truncation in planar MPS decoding."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

from qldpc_fno.artifacts import sha256_file, write_canonical_json
from qldpc_fno.decision.tensor_network import (
    InvalidCosetMassError,
    planar_mps_coset_masses,
)

_SOURCE_PATHS = (Path(__file__), Path(__file__).with_name("tensor_network.py"))
_VALID_MODES = {"columns", "rows"}


def _load_config(path: Path) -> dict[str, object]:
    config = json.loads(path.read_text())
    if config.get("schema_version") != 1:
        raise ValueError("tensor-tolerance schema_version must be 1")
    domain = str(config["seed_domain"])
    expected_seed = int.from_bytes(hashlib.sha256(domain.encode()).digest()[:8], "big")
    if int(config["campaign_seed"]) != expected_seed:
        raise ValueError("campaign_seed must be the SHA-256 derivation of seed_domain")
    tolerances = config.get("tolerances")
    if (
        not isinstance(tolerances, list)
        or not tolerances
        or any(not np.isfinite(value) or float(value) <= 0 for value in tolerances)
        or [float(value) for value in tolerances]
        != sorted({float(value) for value in tolerances})
    ):
        raise ValueError("tolerances must be unique increasing positive finite values")
    modes = config.get("contraction_modes")
    if not isinstance(modes, list) or not modes or not set(modes) <= _VALID_MODES:
        raise ValueError("contraction_modes must be a nonempty subset of columns and rows")
    comparator = config.get("fixed_comparator")
    if not isinstance(comparator, dict):
        raise TypeError("fixed_comparator must be an object")
    if comparator.get("mode") not in _VALID_MODES:
        raise ValueError("fixed_comparator mode must be columns or rows")
    if type(comparator.get("chi")) is not int or int(comparator["chi"]) <= 0:
        raise ValueError("fixed_comparator chi must be a positive integer")
    if float(config.get("log_mass_ratio_tolerance", -1)) <= 0:
        raise ValueError("log_mass_ratio_tolerance must be positive")
    if config.get("logical_class_required") is not True:
        raise ValueError("logical_class_required must be true")
    return config


def _pairwise_log_mass_ratio_error(candidate: np.ndarray, reference: np.ndarray) -> float:
    candidate_logs = np.log(candidate / candidate.sum())
    reference_logs = np.log(reference / reference.sum())
    return max(
        abs(
            (candidate_logs[first] - candidate_logs[second])
            - (reference_logs[first] - reference_logs[second])
        )
        for first in range(4)
        for second in range(first + 1, 4)
    )


def _spectral_diagnostics(work: dict[str, object]) -> dict[str, object]:
    events = work["truncation_events"]
    assert isinstance(events, list)
    spectra = [
        spectrum
        for event in events
        for spectrum in event["spectral_summaries"]
    ]
    retained = [int(spectrum["retained_rank"]) for spectrum in spectra]
    discarded = [
        float(spectrum["discarded_squared_weight_fraction"]) for spectrum in spectra
    ]
    return {
        "truncation_event_count": len(events),
        "local_spectrum_count": len(spectra),
        "retained_rank_min": min(retained) if retained else 0,
        "retained_rank_max": max(retained) if retained else 0,
        "retained_rank_mean": float(np.mean(retained)) if retained else 0.0,
        "maximum_local_discarded_squared_weight_fraction": max(discarded)
        if discarded
        else 0.0,
        "mean_local_discarded_squared_weight_fraction": float(np.mean(discarded))
        if discarded
        else 0.0,
    }


def run_tensor_tolerance_study(
    config_path: Path,
    reference_path: Path,
    output_dir: Path,
) -> dict[str, object]:
    """Evaluate fixed local-spectrum tolerances on a locked discovery table."""
    config = _load_config(config_path)
    reference_sha = sha256_file(reference_path)
    if reference_sha != str(config["expected_reference_sha256"]):
        raise ValueError("reference artifact SHA-256 does not match the config")
    reference = json.loads(reference_path.read_text())
    if reference.get("schema_version") != 1:
        raise ValueError("reference artifact schema_version must be 1")
    instances = reference.get("instances")
    reference_rows = reference.get("rows")
    if not isinstance(instances, list) or not instances:
        raise ValueError("reference artifact has no instances")
    if not isinstance(reference_rows, list):
        raise TypeError("reference artifact has no action rows")

    rows: list[dict[str, object]] = []
    for instance in instances:
        distance = int(instance["distance"])
        error_rate = float(instance["error_rate"])
        syndrome = np.asarray(instance["syndrome"], dtype=np.uint8)
        reference_probabilities = np.asarray(
            instance["reference_probabilities"], dtype=np.float64
        )
        reference_class = int(instance["reference_selected_class"])
        for mode in config["contraction_modes"]:
            for tolerance_value in config["tolerances"]:
                tolerance = float(tolerance_value)
                try:
                    result = planar_mps_coset_masses(
                        rows=distance,
                        columns=distance,
                        syndrome=syndrome,
                        error_rate=error_rate,
                        chi=None,
                        mode=str(mode),
                        tol=tolerance,
                        trace_work=True,
                    )
                    probabilities = result.probabilities
                    work = result.work
                    assert work is not None
                    row = {
                        "instance_id": instance["instance_id"],
                        "distance": distance,
                        "error_rate": error_rate,
                        "syndrome_weight": int(instance["syndrome_weight"]),
                        "mode": mode,
                        "chi": None,
                        "tol": tolerance,
                        "solver_valid": True,
                        "selected_class": result.selected_class,
                        "selected_class_correct": result.selected_class == reference_class,
                        "probabilities": probabilities.tolist(),
                        "total_variation": float(
                            0.5 * np.abs(probabilities - reference_probabilities).sum()
                        ),
                        "maximum_log_mass_ratio_error": float(
                            _pairwise_log_mass_ratio_error(
                                probabilities, reference_probabilities
                            )
                        ),
                        "work": work,
                        **_spectral_diagnostics(work),
                    }
                except InvalidCosetMassError as error:
                    work = error.work
                    assert work is not None
                    row = {
                        "instance_id": instance["instance_id"],
                        "distance": distance,
                        "error_rate": error_rate,
                        "syndrome_weight": int(instance["syndrome_weight"]),
                        "mode": mode,
                        "chi": None,
                        "tol": tolerance,
                        "solver_valid": False,
                        "selected_class": None,
                        "selected_class_correct": False,
                        "probabilities": None,
                        "raw_masses": error.masses.tolist(),
                        "total_variation": None,
                        "maximum_log_mass_ratio_error": None,
                        "work": work,
                        **_spectral_diagnostics(work),
                    }
                rows.append(row)

    comparator_config = config["fixed_comparator"]
    assert isinstance(comparator_config, dict)
    comparator_rows = [
        row
        for row in reference_rows
        if row["mode"] == comparator_config["mode"]
        and int(row["chi"]) == int(comparator_config["chi"])
    ]
    if len(comparator_rows) != len(instances):
        raise ValueError("reference artifact lacks one fixed comparator row per instance")
    comparator_work = sum(
        int(row["work"]["estimated_arithmetic_flops"]) for row in comparator_rows
    )
    fixed_comparator = {
        **comparator_config,
        "instance_count": len(comparator_rows),
        "solver_invalid_count": sum(not bool(row["solver_valid"]) for row in comparator_rows),
        "logical_class_failure_count": sum(
            not bool(row["selected_class_correct"]) for row in comparator_rows
        ),
        "estimated_arithmetic_flops": comparator_work,
    }

    action_summaries: list[dict[str, object]] = []
    for mode in config["contraction_modes"]:
        for tolerance_value in config["tolerances"]:
            tolerance = float(tolerance_value)
            selected = [
                row for row in rows if row["mode"] == mode and row["tol"] == tolerance
            ]
            total_work = sum(
                int(row["work"]["estimated_arithmetic_flops"]) for row in selected
            )
            action_summaries.append(
                {
                    "mode": mode,
                    "tol": tolerance,
                    "instance_count": len(selected),
                    "solver_invalid_count": sum(
                        not bool(row["solver_valid"]) for row in selected
                    ),
                    "logical_class_failure_count": sum(
                        not bool(row["selected_class_correct"]) for row in selected
                    ),
                    "mass_fidelity_failure_count": sum(
                        not bool(row["solver_valid"])
                        or float(row["maximum_log_mass_ratio_error"])
                        > float(config["log_mass_ratio_tolerance"])
                        for row in selected
                    ),
                    "estimated_arithmetic_flops": total_work,
                    "work_savings_fraction_vs_fixed_comparator": (
                        comparator_work - total_work
                    )
                    / comparator_work,
                    "maximum_retained_rank": max(
                        int(row["retained_rank_max"]) for row in selected
                    ),
                    "maximum_local_discarded_squared_weight_fraction": max(
                        float(row["maximum_local_discarded_squared_weight_fraction"])
                        for row in selected
                    ),
                }
            )
    safe = [
        summary
        for summary in action_summaries
        if int(summary["solver_invalid_count"]) == 0
        and int(summary["logical_class_failure_count"]) == 0
    ]
    best_safe = (
        None
        if not safe
        else min(
            safe,
            key=lambda summary: (
                int(summary["estimated_arithmetic_flops"]),
                str(summary["mode"]),
                float(summary["tol"]),
            ),
        )
    )
    run_label = "discovery_nonconfirmatory" if len(instances) == 96 else "reduced_non_scientific"
    payload: dict[str, object] = {
        "schema_version": 1,
        "run_label": run_label,
        "reference_sha256": reference_sha,
        "config": config,
        "instance_count": len(instances),
        "rows": rows,
        "action_summaries": action_summaries,
        "fixed_comparator": fixed_comparator,
        "best_globally_class_safe_tolerance_action": best_safe,
        "claim_scope": {
            "confirmation": False,
            "latency": False,
            "local_discarded_weight_is_global_error_bound": False,
            "rl_or_world_model": False,
        },
        "provenance": {
            "config_path": str(config_path),
            "config_sha256": sha256_file(config_path),
            "reference_path": str(reference_path),
            "reference_sha256": reference_sha,
            "source_sha256": {
                str(path): sha256_file(path) for path in _SOURCE_PATHS
            },
        },
    }
    output_dir.mkdir(parents=True, exist_ok=False)
    write_canonical_json(output_dir / "tensor_tolerance_discovery.json", payload)
    return payload
