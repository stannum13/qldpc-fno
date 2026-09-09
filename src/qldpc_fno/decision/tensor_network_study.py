"""Paired bond-dimension and contraction-order study for planar MPS decoding."""

from __future__ import annotations

import hashlib
import json
from itertools import pairwise
from pathlib import Path
from time import perf_counter

import numpy as np
from qecsim.models.generic import DepolarizingErrorModel
from qecsim.models.planar import PlanarCode

from qldpc_fno.artifacts import sha256_file, write_canonical_json
from qldpc_fno.decision.tensor_network import (
    QEC_SIM_REFERENCE_COMMIT,
    QEC_SIM_VERSION,
    InvalidCosetMassError,
    exact_planar_coset_masses,
    planar_mps_coset_masses,
)

_SOURCE_PATHS = (Path(__file__), Path(__file__).with_name("tensor_network.py"))
_VALID_MODES = {"columns", "rows", "average"}
_CANONICAL_CONFIG: dict[str, object] = {
    "schema_version": 1,
    "seed_domain": "qldpc-fno/tensor-network-reference/v1",
    "campaign_seed": 6550373459000682169,
    "code_distances": [3, 5],
    "error_rates": [0.05, 0.1, 0.15],
    "instances_per_point": 16,
    "chi_ladder": [1, 2, 4, 8, 16],
    "contraction_modes": ["columns", "rows", "average"],
    "unrestricted_reference_max_distance": 5,
    "high_chi_reference": 32,
    "timing_warmups": 1,
    "timing_repetitions": 3,
    "log_mass_ratio_tolerance": 0.05,
    "log_mass_ratio_tolerance_sweep": [0.001, 0.01, 0.05, 0.1, 0.5, 1.0],
    "minimum_oracle_work_savings_fraction": 0.05,
}


def _load_config(path: Path) -> dict[str, object]:
    config = json.loads(path.read_text())
    if config.get("schema_version") != 1:
        raise ValueError("tensor-network schema_version must be 1")
    domain = str(config["seed_domain"])
    expected_seed = int.from_bytes(hashlib.sha256(domain.encode()).digest()[:8], "big")
    if int(config["campaign_seed"]) != expected_seed:
        raise ValueError("campaign_seed must be the SHA-256 derivation of seed_domain")
    distances = config.get("code_distances")
    if not isinstance(distances, list) or not distances or any(
        type(value) is not int or value < 2 for value in distances
    ):
        raise ValueError("code_distances must be a nonempty list of integers >= 2")
    chi_ladder = config.get("chi_ladder")
    if (
        not isinstance(chi_ladder, list)
        or not chi_ladder
        or any(type(value) is not int or value <= 0 for value in chi_ladder)
        or chi_ladder != sorted(set(chi_ladder))
    ):
        raise ValueError("chi_ladder must contain unique increasing positive integers")
    modes = config.get("contraction_modes")
    if not isinstance(modes, list) or not modes or not set(modes) <= _VALID_MODES:
        raise ValueError("contraction_modes contains an unsupported mode")
    rates = np.asarray(config.get("error_rates"), dtype=np.float64)
    if rates.ndim != 1 or not len(rates) or np.any((rates <= 0) | (rates >= 1)):
        raise ValueError("error_rates must be a nonempty list with entries in (0, 1)")
    for key in ("instances_per_point", "timing_repetitions"):
        if type(config.get(key)) is not int or int(config[key]) <= 0:
            raise ValueError(f"{key} must be a positive integer")
    if type(config.get("timing_warmups")) is not int or int(config["timing_warmups"]) < 0:
        raise ValueError("timing_warmups must be a nonnegative integer")
    savings = float(config.get("minimum_oracle_work_savings_fraction", -1))
    if not 0 <= savings < 1:
        raise ValueError("minimum_oracle_work_savings_fraction must lie in [0, 1)")
    tolerance = float(config.get("log_mass_ratio_tolerance", -1))
    sweep = config.get("log_mass_ratio_tolerance_sweep")
    if (
        not isinstance(sweep, list)
        or not sweep
        or any(float(value) <= 0 for value in sweep)
        or [float(value) for value in sweep] != sorted({float(value) for value in sweep})
        or tolerance not in [float(value) for value in sweep]
    ):
        raise ValueError(
            "log_mass_ratio_tolerance_sweep must be unique, increasing, positive, "
            "and contain the primary tolerance"
        )
    return config


def _seed(campaign_seed: int, distance: int, error_rate: float, index: int) -> int:
    identity = (
        f"qldpc-fno/tensor-network/instance/v1:{campaign_seed}:"
        f"{distance}:{error_rate:.17g}:{index}"
    )
    return int.from_bytes(hashlib.sha256(identity.encode()).digest()[:8], "big")


def _pairwise_log_mass_ratio_error(candidate: np.ndarray, reference: np.ndarray) -> float:
    candidate_logs = np.log(candidate / candidate.sum())
    reference_logs = np.log(reference / reference.sum())
    maximum = 0.0
    for first in range(4):
        for second in range(first + 1, 4):
            error = abs(
                (candidate_logs[first] - candidate_logs[second])
                - (reference_logs[first] - reference_logs[second])
            )
            maximum = max(maximum, float(error))
    return maximum


def _time_candidate(
    *,
    distance: int,
    syndrome: np.ndarray,
    error_rate: float,
    chi: int,
    mode: str,
    warmups: int,
    repetitions: int,
) -> tuple[object, list[float]]:
    kwargs = {
        "rows": distance,
        "columns": distance,
        "syndrome": syndrome,
        "error_rate": error_rate,
        "chi": chi,
        "mode": mode,
    }
    traced = planar_mps_coset_masses(**kwargs, trace_work=True)
    for _ in range(warmups):
        planar_mps_coset_masses(**kwargs)
    times: list[float] = []
    for _ in range(repetitions):
        started = perf_counter()
        planar_mps_coset_masses(**kwargs)
        times.append(perf_counter() - started)
    return traced, times


def _summaries(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    keys = tuple(
        dict.fromkeys(
            (int(row["distance"]), float(row["error_rate"]), str(row["mode"]), int(row["chi"]))
            for row in rows
        )
    )
    summaries: list[dict[str, object]] = []
    for distance, error_rate, mode, chi in keys:
        selected = [
            row
            for row in rows
            if row["distance"] == distance
            and row["error_rate"] == error_rate
            and row["mode"] == mode
            and row["chi"] == chi
        ]
        valid = [row for row in selected if bool(row["solver_valid"])]
        wall_times = [value for row in valid for value in row["untraced_wall_seconds"]]
        summaries.append(
            {
                "distance": distance,
                "error_rate": error_rate,
                "mode": mode,
                "chi": chi,
                "instance_count": len(selected),
                "solver_valid_fraction": len(valid) / len(selected),
                "selected_class_agreement": float(
                    np.mean([bool(row["selected_class_correct"]) for row in selected])
                ),
                "mean_total_variation_over_valid": (
                    None
                    if not valid
                    else float(np.mean([float(row["total_variation"]) for row in valid]))
                ),
                "maximum_log_mass_ratio_error_over_valid": (
                    None
                    if not valid
                    else max(float(row["maximum_log_mass_ratio_error"]) for row in valid)
                ),
                "median_untraced_wall_seconds": (
                    None if not wall_times else float(np.median(wall_times))
                ),
                "mean_estimated_arithmetic_flops": float(
                    np.mean(
                        [float(row["work"]["estimated_arithmetic_flops"]) for row in selected]
                    )
                ),
                "maximum_peak_observed_array_elements": max(
                    int(row["work"]["peak_observed_array_elements"]) for row in selected
                ),
            }
        )
    return summaries


def _adaptive_and_monotonicity_audits(
    rows: list[dict[str, object]], tolerance: float, minimum_savings_fraction: float
) -> tuple[dict[str, object], dict[str, object], str]:
    sequences: dict[tuple[str, str], list[dict[str, object]]] = {}
    for row in rows:
        sequences.setdefault((str(row["instance_id"]), str(row["mode"])), []).append(row)

    nonmonotonic_tv = 0
    nonmonotonic_log_ratio = 0
    for sequence in sequences.values():
        ordered = sorted(sequence, key=lambda row: int(row["chi"]))
        valid_ordered = [row for row in ordered if bool(row["solver_valid"])]
        nonmonotonic_tv += int(
            any(
                float(right["total_variation"]) > float(left["total_variation"]) + 1e-12
                for left, right in pairwise(valid_ordered)
            )
        )
        nonmonotonic_log_ratio += int(
            any(
                float(right["maximum_log_mass_ratio_error"])
                > float(left["maximum_log_mass_ratio_error"]) + 1e-12
                for left, right in pairwise(valid_ordered)
            )
        )

    by_instance: dict[str, list[dict[str, object]]] = {}
    for row in rows:
        by_instance.setdefault(str(row["instance_id"]), []).append(row)

    def feasible(row: dict[str, object]) -> bool:
        return (
            bool(row["solver_valid"])
            and bool(row["selected_class_correct"])
            and float(row["maximum_log_mass_ratio_error"]) <= tolerance
        )

    action_ids = sorted({(str(row["mode"]), int(row["chi"])) for row in rows})
    globally_feasible: list[tuple[tuple[str, int], int]] = []
    for action in action_ids:
        action_rows = [
            next(
                row
                for row in instance_rows
                if (str(row["mode"]), int(row["chi"])) == action
            )
            for instance_rows in by_instance.values()
        ]
        if all(feasible(row) for row in action_rows):
            globally_feasible.append(
                (
                    action,
                    sum(int(row["work"]["estimated_arithmetic_flops"]) for row in action_rows),
                )
            )

    oracle_rows: list[dict[str, object]] = []
    for instance_id, instance_rows in by_instance.items():
        candidates = [row for row in instance_rows if feasible(row)]
        if not candidates:
            oracle_rows.append({"instance_id": instance_id, "action": None, "work": None})
            continue
        best = min(
            candidates,
            key=lambda row: (
                int(row["work"]["estimated_arithmetic_flops"]),
                str(row["mode"]),
                int(row["chi"]),
            ),
        )
        oracle_rows.append(
            {
                "instance_id": instance_id,
                "action": {"mode": str(best["mode"]), "chi": int(best["chi"])},
                "work": int(best["work"]["estimated_arithmetic_flops"]),
            }
        )

    best_fixed = min(globally_feasible, key=lambda item: (item[1], item[0])) if globally_feasible else None
    oracle_complete = all(row["work"] is not None for row in oracle_rows)
    if best_fixed is not None and oracle_complete:
        fixed_work = best_fixed[1]
        oracle_work = sum(int(row["work"]) for row in oracle_rows)
        savings_fraction = (fixed_work - oracle_work) / fixed_work
    else:
        fixed_work = oracle_work = None
        savings_fraction = None
    oracle_actions = {
        None if row["action"] is None else (row["action"]["mode"], row["action"]["chi"])
        for row in oracle_rows
    }
    adaptive = {
        "criterion": (
            "actions require the reference logical class and maximum pairwise "
            f"log-mass-ratio error <= {tolerance}"
        ),
        "work_measure": "estimated_arithmetic_flops",
        "minimum_oracle_work_savings_fraction": minimum_savings_fraction,
        "best_fixed_action": (
            None
            if best_fixed is None
            else {
                "mode": best_fixed[0][0],
                "chi": best_fixed[0][1],
                "total_work": fixed_work,
            }
        ),
        "per_instance_oracle": oracle_rows,
        "oracle_total_work": oracle_work,
        "oracle_work_savings_fraction": savings_fraction,
        "distinct_oracle_actions": len(oracle_actions),
        "instance_dependent": (
            savings_fraction is not None
            and savings_fraction >= minimum_savings_fraction
            and len(oracle_actions) > 1
        ),
    }
    monotonicity = {
        "tested_sequences": len(sequences),
        "total_variation_nonmonotonic_sequence_count": nonmonotonic_tv,
        "log_mass_ratio_nonmonotonic_sequence_count": nonmonotonic_log_ratio,
        "monotonicity_assumed": False,
    }
    if best_fixed is None or not oracle_complete:
        status = "unresolved_accuracy_ladder_has_no_complete_comparator"
    elif adaptive["instance_dependent"]:
        status = "open_instance_dependent_accuracy_work"
    else:
        status = "closed_no_meaningful_instance_variation"
    return adaptive, monotonicity, status


def run_tensor_network_study(config_path: Path, output_dir: Path) -> dict[str, object]:
    """Run a paired planar-code MPS accuracy/work sweep."""
    config = _load_config(config_path)
    model = DepolarizingErrorModel()
    instances: list[dict[str, object]] = []
    rows: list[dict[str, object]] = []
    enumeration_validation_errors: list[float] = []

    for distance in config["code_distances"]:
        code = PlanarCode(int(distance), int(distance))
        for error_rate_value in config["error_rates"]:
            error_rate = float(error_rate_value)
            for instance_index in range(int(config["instances_per_point"])):
                seed = _seed(
                    int(config["campaign_seed"]), int(distance), error_rate, instance_index
                )
                error = model.generate(code, error_rate, np.random.default_rng(seed))
                syndrome = np.asarray(error @ code.stabilizers.T % 2, dtype=np.uint8)
                instance_id = f"d{distance}/p{error_rate:.6f}/i{instance_index:04d}"
                if int(distance) <= int(config["unrestricted_reference_max_distance"]):
                    reference_chi = None
                    reference_kind = "unrestricted_mps_exact_up_to_roundoff"
                else:
                    reference_chi = int(config["high_chi_reference"])
                    reference_kind = "finite_high_chi_mps_not_certified_exact"
                reference = planar_mps_coset_masses(
                    rows=int(distance),
                    columns=int(distance),
                    syndrome=syndrome,
                    error_rate=error_rate,
                    chi=reference_chi,
                    mode="average",
                )
                if int(distance) == 3:
                    enumerated = exact_planar_coset_masses(
                        rows=3,
                        columns=3,
                        syndrome=syndrome,
                        error_rate=error_rate,
                    )
                    enumeration_validation_errors.append(
                        float(np.max(np.abs(reference.probabilities - enumerated.probabilities)))
                    )
                instances.append(
                    {
                        "instance_id": instance_id,
                        "distance": int(distance),
                        "error_rate": error_rate,
                        "instance_index": instance_index,
                        "sampler_seed": seed,
                        "syndrome": syndrome.tolist(),
                        "syndrome_weight": int(syndrome.sum()),
                        "reference_kind": reference_kind,
                        "reference_chi": reference_chi,
                        "reference_probabilities": reference.probabilities.tolist(),
                        "reference_selected_class": reference.selected_class,
                    }
                )
                for mode in config["contraction_modes"]:
                    for chi in config["chi_ladder"]:
                        try:
                            candidate, wall_times = _time_candidate(
                                distance=int(distance),
                                syndrome=syndrome,
                                error_rate=error_rate,
                                chi=int(chi),
                                mode=str(mode),
                                warmups=int(config["timing_warmups"]),
                                repetitions=int(config["timing_repetitions"]),
                            )
                            solver_valid = True
                            raw_masses = candidate.masses.tolist()
                            candidate_probabilities = candidate.probabilities
                            selected_class = candidate.selected_class
                            work = candidate.work
                            total_variation = float(
                                0.5
                                * np.abs(candidate_probabilities - reference.probabilities).sum()
                            )
                            log_ratio_error = _pairwise_log_mass_ratio_error(
                                candidate.masses, reference.masses
                            )
                        except InvalidCosetMassError as failure:
                            solver_valid = False
                            raw_masses = failure.masses.tolist()
                            candidate_probabilities = None
                            selected_class = None
                            work = failure.work
                            wall_times = []
                            total_variation = None
                            log_ratio_error = None
                        rows.append(
                            {
                                "instance_id": instance_id,
                                "distance": int(distance),
                                "error_rate": error_rate,
                                "syndrome": syndrome.tolist(),
                                "syndrome_weight": int(syndrome.sum()),
                                "reference_kind": reference_kind,
                                "reference_chi": reference_chi,
                                "reference_selected_class": reference.selected_class,
                                "reference_probabilities": reference.probabilities.tolist(),
                                "mode": str(mode),
                                "chi": int(chi),
                                "solver_valid": solver_valid,
                                "raw_masses": raw_masses,
                                "selected_class": selected_class,
                                "selected_class_correct": (
                                    solver_valid and selected_class == reference.selected_class
                                ),
                                "probabilities": (
                                    None
                                    if candidate_probabilities is None
                                    else candidate_probabilities.tolist()
                                ),
                                "total_variation": total_variation,
                                "maximum_log_mass_ratio_error": log_ratio_error,
                                "work": work,
                                "untraced_wall_seconds": wall_times,
                            }
                        )

    tolerance = float(config["log_mass_ratio_tolerance"])
    adaptive, monotonicity, gate_status = _adaptive_and_monotonicity_audits(
        rows,
        tolerance,
        float(config["minimum_oracle_work_savings_fraction"]),
    )
    criterion_sensitivity: list[dict[str, object]] = []
    for sensitivity_tolerance in config["log_mass_ratio_tolerance_sweep"]:
        sensitivity, _, sensitivity_status = _adaptive_and_monotonicity_audits(
            rows,
            float(sensitivity_tolerance),
            float(config["minimum_oracle_work_savings_fraction"]),
        )
        criterion_sensitivity.append(
            {
                "tolerance": float(sensitivity_tolerance),
                "status": sensitivity_status,
                "best_fixed_action": sensitivity["best_fixed_action"],
                "oracle_total_work": sensitivity["oracle_total_work"],
                "oracle_work_savings_fraction": sensitivity[
                    "oracle_work_savings_fraction"
                ],
                "distinct_oracle_actions": sensitivity["distinct_oracle_actions"],
            }
        )
    canonical_shape = config == _CANONICAL_CONFIG
    if not canonical_shape:
        gate_status = "unresolved_reduced_non_scientific"
    payload: dict[str, object] = {
        "schema_version": 1,
        "run_label": "canonical_discovery" if canonical_shape else "reduced_non_scientific",
        "gate5_status": gate_status,
        "claim_scope": {
            "noise": "code-capacity iid depolarizing noise with perfect syndrome extraction",
            "code_family": "unrotated planar surface code",
            "latency": (
                "single-process host wall time; not an FPGA, streaming, or deployment latency claim"
            ),
            "work": (
                "estimated arithmetic FLOPs from NumPy einsum paths plus stated dense QR/SVD "
                "formulae; not hardware instructions, FPGA cycles, or bytes"
            ),
        },
        "config": config,
        "reference": {
            "implementation": "qecsim PlanarMPSDecoder",
            "qecsim_version": QEC_SIM_VERSION,
            "audited_upstream_commit": QEC_SIM_REFERENCE_COMMIT,
            "small_distance_independent_enumeration_max_probability_error": (
                max(enumeration_validation_errors) if enumeration_validation_errors else None
            ),
            "finite_high_chi_is_certified_exact": False,
        },
        "instances": instances,
        "rows": rows,
        "summaries": _summaries(rows),
        "adaptive_signal": adaptive,
        "criterion_sensitivity": criterion_sensitivity,
        "monotonicity_audit": monotonicity,
        "provenance": {
            "config_path": str(config_path),
            "config_sha256": sha256_file(config_path),
            "source_sha256": {str(path): sha256_file(path) for path in _SOURCE_PATHS},
        },
    }
    output_path = output_dir / "tensor_network_reference.json"
    write_canonical_json(output_path, payload)
    return payload
